import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from loguru import logger as logging
from tqdm import tqdm

from stable_worldmodel.utils import DEFAULT_CACHE_DIR, HF_BASE_URL


def get_cache_dir(
    override_root: Path | None = None,
    sub_folder: str | None = None,
) -> Path:
    base = override_root
    if override_root is None:
        base = os.getenv('STABLEWM_HOME', str(DEFAULT_CACHE_DIR))

    cache_path = (
        Path(base, sub_folder) if sub_folder is not None else Path(base)
    )

    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def ensure_dir_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def filter_moving_starts(
    dataset,
    valid_indices,
    offset,
    thresh,
    pos_col='privileged/block_0_pos',
    episode_col='episode_idx',
):
    """Keep start rows whose tracked position MOVES over the goal horizon.

    De-trivialises offset-goal eval / metric on manipulation tasks where the
    object is stationary in most windows (e.g. OGBench-Cube: at offset 25 the
    cube is already within the 0.04 m success threshold of its 25-ahead pose in
    ~71% of windows, so random actions ``succeed``). A candidate start row ``r``
    (a global dataset row) qualifies iff the tracked position moves at least
    ``thresh`` over ``offset`` steps WITHIN the same episode::

        ||pos[r + offset] - pos[r]|| >= thresh   and   episode(r+offset)==episode(r)

    ``pos_col`` must be the SAME privileged position the task's success
    criterion uses (the cube block position, ``privileged/block_0_pos``), so the
    surviving windows are exactly those that demand real manipulation to reach
    the goal -- and so eval and the GCS metric, fed the same ``valid_indices``,
    seed and threshold, score an identical start set.

    Rows are stored episode-contiguous and step-ordered (global row =
    ``episode_offset + step_idx``), so the row ``offset`` steps ahead in the
    same episode is ``r + offset``; the episode-id guard keeps the lookup from
    crossing an episode boundary (it is always satisfied for rows already
    restricted to ``step_idx <= episode_len - offset - 1``, but is kept explicit
    so the filter is correct on its own).

    Args:
        dataset: swm dataset reader (provides ``get_col_data``).
        valid_indices: 1-D int array of candidate global start rows.
        offset: goal horizon in steps.
        thresh: minimum L2 displacement (same units as ``pos_col``).
        pos_col: column holding the tracked position (3-D for the cube).
        episode_col: episode-index column name.

    Returns:
        The subset of ``valid_indices`` (ascending order preserved) that qualify.
    """
    valid_indices = np.asarray(valid_indices)
    pos = np.asarray(dataset.get_col_data(pos_col), dtype=np.float64)
    ep = np.asarray(dataset.get_col_data(episode_col))
    ahead = valid_indices + int(offset)
    in_ep = (ahead < len(pos)) & (
        ep[np.minimum(ahead, len(pos) - 1)] == ep[valid_indices]
    )
    disp = np.full(len(valid_indices), -np.inf, dtype=np.float64)
    disp[in_ep] = np.linalg.norm(
        pos[ahead[in_ep]] - pos[valid_indices[in_ep]], axis=1
    )
    return valid_indices[disp >= float(thresh)]


def load_dataset(
    name: str,
    cache_dir: str = None,
    format: str | None = None,
    **kwargs,
):
    """Resolve a dataset name to a local path and dispatch to the matching
    format reader from the registry.

    Supported names:

    1. **Local path** — file or directory.
    2. **HuggingFace repo** (``<user>/<repo>``) — downloaded and cached under
       ``<cache_dir>/datasets/<user>--<repo>/``.
    3. **Format scheme** (e.g. ``lerobot://lerobot/pusht``) — passed through
       to the matching format unchanged.

    The format is auto-detected via :func:`detect_format` unless ``format`` is
    provided explicitly. To register a new format, decorate a
    :class:`~stable_worldmodel.data.format.Format` subclass with
    :func:`~stable_worldmodel.data.format.register_format`.

    Args:
        name: Local path, HF repo id, or scheme-prefixed identifier.
        cache_dir: Root cache directory. Defaults to ``STABLEWM_HOME`` or
            ``~/.stable_worldmodel``.
        format: Explicit format name (skips detection).
        **kwargs: Forwarded to the format's reader.

    Returns:
        A reader instance (typically a
        :class:`~stable_worldmodel.data.dataset.Dataset` subclass).
    """
    from stable_worldmodel.data.format import (
        FORMATS,
        detect_format,
        get_format,
    )

    name = str(name)

    # Scheme-prefixed identifiers (e.g. lerobot://...) bypass path resolution.
    if '://' in name:
        if format is None:
            for fmt in FORMATS.values():
                if fmt.detect(name):
                    return fmt.open_reader(name, **kwargs)
            raise ValueError(f'No format detected for {name!r}')
        return get_format(format).open_reader(name, **kwargs)

    datasets_dir = get_cache_dir(cache_dir, sub_folder='datasets')
    ensure_dir_exists(datasets_dir)
    path = _resolve_dataset(name, datasets_dir)

    if format is not None:
        return get_format(format).open_reader(path, **kwargs)

    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(
            f'No format detected for {path!r}; pass format= explicitly.'
        )
    return fmt.open_reader(path, **kwargs)


def _resolve_dataset(name: str, datasets_dir: Path) -> Path:
    """Resolve *name* (local path or HF repo id) to a local path.

    Returns whatever exists on disk — file or directory. Format detection
    happens after this in :func:`load_dataset`. Local layout for cached
    datasets is a directory under ``<datasets_dir>/<name>/``; the directory
    may hold a ``foo.lance/`` table, a ``foo.h5`` file, or any other
    layout a registered format can detect.
    """
    local = Path(name)
    if not local.is_absolute():
        local = datasets_dir / local

    if local.exists():
        return local

    # HuggingFace repo: <user>/<repo>
    if '/' in name and not name.startswith(('.', '/')):
        return _resolve_dataset_hf(name, datasets_dir)

    raise FileNotFoundError(
        f'Cannot resolve {name!r}: not a local path or HF repo id.'
    )


# Suffixes we recognise on HF: a `.lance` directory (preferred) or a
# `.h5` / `.hdf5` file. Each format is downloaded in its native shape — no
# tar/zst wrapping.
_HF_FILE_SUFFIXES: tuple[str, ...] = ('.h5', '.hdf5')
_HF_DIR_SUFFIXES: tuple[str, ...] = ('.lance',)


def _hf_list_tree(repo_id: str, sub_path: str = '') -> list[dict]:
    """One HF API call: list entries at ``<repo>/tree/main/<sub_path>``."""
    suffix = f'/{sub_path}' if sub_path else ''
    api_url = f'{HF_BASE_URL}/api/datasets/{repo_id}/tree/main{suffix}'
    with urllib.request.urlopen(api_url) as resp:
        return json.loads(resp.read())


def _hf_find_dataset_entry(repo_id: str) -> dict:
    """Return the first top-level entry that looks like a dataset.

    Preference: a ``*.lance`` directory wins over an ``*.h5`` file when
    both are present, since lance is the default format.
    """
    entries = _hf_list_tree(repo_id)

    for entry in entries:
        path = entry.get('path', '')
        if entry.get('type') == 'directory' and path.endswith(
            _HF_DIR_SUFFIXES
        ):
            return entry
    for entry in entries:
        path = entry.get('path', '')
        if entry.get('type') == 'file' and path.endswith(_HF_FILE_SUFFIXES):
            return entry

    raise FileNotFoundError(
        f'No dataset found in HF repo {repo_id}: expected a top-level '
        f'`*.lance` directory or `*.h5`/`*.hdf5` file.'
    )


def _hf_walk_files(repo_id: str, sub_path: str) -> list[str]:
    """Recursively list every *file* path under ``sub_path`` on HF."""
    out: list[str] = []
    stack = [sub_path]
    while stack:
        current = stack.pop()
        for entry in _hf_list_tree(repo_id, current):
            path = entry.get('path', '')
            if entry.get('type') == 'directory':
                stack.append(path)
            else:
                out.append(path)
    return out


def _resolve_dataset_hf(repo_id: str, datasets_dir: Path) -> Path:
    """Resolve a HF repo id, downloading on first use.

    Local layout: ``<datasets_dir>/<user>--<repo>/`` — the directory is
    returned as-is and format detection picks up whatever lives inside
    (``*.lance``, ``*.h5``, …).
    """
    local_dir = datasets_dir / repo_id.replace('/', '--')

    if local_dir.is_dir() and any(local_dir.iterdir()):
        logging.info(f'Using cached dataset for {repo_id} at {local_dir}')
        return local_dir

    logging.info(f'Downloading dataset {repo_id} from HuggingFace...')
    local_dir.mkdir(parents=True, exist_ok=True)

    entry = _hf_find_dataset_entry(repo_id)
    entry_path = entry['path']

    if entry.get('type') == 'directory':
        files = _hf_walk_files(repo_id, entry_path)
        if not files:
            raise FileNotFoundError(
                f"HF repo {repo_id}: directory '{entry_path}' is empty."
            )
        for remote in tqdm(files, desc=f'Fetching {entry_path}'):
            url = f'{HF_BASE_URL}/datasets/{repo_id}/resolve/main/{remote}'
            dest = local_dir / remote
            dest.parent.mkdir(parents=True, exist_ok=True)
            _download(url, dest)
    else:
        url = f'{HF_BASE_URL}/datasets/{repo_id}/resolve/main/{entry_path}'
        dest = local_dir / entry_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f'Fetching {url}')
        _download(url, dest)

    return local_dir


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    response = urllib.request.urlopen(url)
    total = int(response.headers.get('Content-Length', 0)) or None
    with (
        open(dest, 'wb') as f,
        tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as bar,
    ):
        chunk = response.read(8192)
        while chunk:
            f.write(chunk)
            bar.update(len(chunk))
            chunk = response.read(8192)


def convert(
    source,
    dest,
    *,
    source_format: str | None = None,
    dest_format: str = 'lance',
    cache_dir: str | None = None,
    progress: bool = True,
    **dest_kwargs,
) -> None:
    """Convert a dataset from one registered format to another.

    Reads each episode from *source* and writes it through the writer of
    *dest_format*. Format detection follows the same rules as
    :func:`load_dataset` — autodetect by default, or pass ``source_format``
    explicitly.

    Args:
        source: Path or identifier accepted by :func:`load_dataset`.
        dest: Output path for the destination writer.
        source_format: Force a source format (skips detection).
        dest_format: Registered writer name (default ``'lance'``).
        cache_dir: Forwarded to the source loader for HF/local resolution.
        progress: Show a progress bar over episodes.
        **dest_kwargs: Forwarded to the destination writer.

    Example::

        from stable_worldmodel.data import convert
        convert('data.lance', 'data_video', dest_format='video')
    """
    from stable_worldmodel.data.format import get_format

    src = load_dataset(source, cache_dir=cache_dir, format=source_format)
    writer_cls = get_format(dest_format)

    iterator = range(len(src.lengths))
    if progress:
        iterator = tqdm(iterator, desc=f'Converting → {dest_format}')

    def episodes():
        for ep_idx in iterator:
            ep = src.load_episode(ep_idx)
            yield _episode_to_step_lists(ep, int(src.lengths[ep_idx]))

    with writer_cls.open_writer(dest, **dest_kwargs) as writer:
        writer.write_episodes(episodes())


def _episode_to_step_lists(ep: dict, ep_len: int) -> dict[str, list]:
    """Adapt an episode dict from a reader to the ``{col: [step_arr, ...]}``
    shape that writers consume.

    Specifically:
      - Tensors → NumPy arrays.
      - Image arrays in ``(N, C, H, W)`` are transposed back to ``(N, H, W, C)``.
      - Image arrays in float dtypes (e.g. LeRobot's ``ToTensor``-normalised
        ``[0, 1]`` floats) are rescaled to ``uint8 [0, 255]`` so downstream
        writers (Lance JPEG encode, Video MP4 encode, HDF5 fixed-dtype
        datasets) receive a consistent display-range integer image.
      - Scalars (e.g. flattened string columns) are repeated ``ep_len`` times.
    """
    out: dict[str, list] = {}
    for col, val in ep.items():
        if isinstance(val, torch.Tensor):
            arr = val.detach().cpu().numpy()
        elif isinstance(val, np.ndarray):
            arr = val
        else:
            out[col] = [val] * ep_len
            continue

        if arr.ndim == 4 and arr.shape[1] in (1, 3):
            arr = arr.transpose(0, 2, 3, 1)

        # Float image → uint8. LeRobot's ToTensor pipeline produces float32
        # in [0, 1]; HDF5 / Lance / Video tworoom-style readers all assume
        # uint8 HxWxC. Detect by shape (3D HWC or 4D NHWC with 1/3 channels)
        # and float dtype, then clip-and-scale.
        if (
            arr.dtype.kind == 'f'
            and arr.ndim in (3, 4)
            and arr.shape[-1] in (1, 3)
        ):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

        out[col] = list(arr)
    return out


from stable_worldmodel.data.normalization import (  # noqa: E402
    IdentityScaler,
    PercentileScaler,
    ZScoreScaler,
    get_scaler,
)


def column_normalizer(
    dataset, source: str, target: str, method: str = 'zscore'
):
    """Build a per-column normalizer :class:`WrapTorchTransform` from dataset stats.

    Args:
        dataset: A dataset exposing ``get_col_data(col)`` returning an array.
        source: Column name to read.
        target: Column name to write.
        method: One of ``'zscore'`` (default), ``'percentile'``, or ``'none'``.
            ``'none'`` returns a pass-through identity transform so call sites
            can stay uniform.

    Returns:
        A picklable :class:`WrapTorchTransform` wrapping a fitted scaler.
    """
    # Lazy import — stable_pretraining is a training-only dep.
    from stable_pretraining.data.transforms import WrapTorchTransform

    scaler = get_scaler(method)
    if method != 'none':
        data = np.array(dataset.get_col_data(source))
        scaler.fit(data)
    return WrapTorchTransform(scaler, source=source, target=target)


__all__ = [
    'load_dataset',
    'convert',
    'get_cache_dir',
    'ensure_dir_exists',
    'filter_moving_starts',
    'IdentityScaler',
    'PercentileScaler',
    'ZScoreScaler',
    'column_normalizer',
    'get_scaler',
]
