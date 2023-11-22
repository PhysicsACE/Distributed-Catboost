import glob
import uuid
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from ray.actor import ActorHandle

try:
    import cupy as cp
except ImportError:
    cp = None

import os

import numpy as np
import sys
import pandas as pd
import ray
from ray import logger
from ray.util.annotations import DeveloperAPI, PublicAPI

from xgboost_ray.data_sources import DataSource, RayFileType, data_sources
from xgboost_ray.matrix import RayShardingMode, _get_sharding_indices, concat_dataframes

try:
    from ray.data.dataset import Dataset as RayDataset
except ImportError:

    class RayDataset:
        pass

Data = Union[str, List[str], np.ndarray, pd.DataFrame, pd.Series]

def _can_load_distributed(source: Data) -> bool:
    """Returns True if it might be possible to use distributed data loading"""
    from xgboost_ray.data_sources.modin import Modin

    if isinstance(source, (int, float, bool)):
        return False
    elif Modin.is_data_type(source):
        return True
    elif isinstance(source, RayDataset):
        return True
    elif isinstance(source, str):
        # Strings should point to files or URLs
        # Usually parquet files point to directories
        return source.endswith(".parquet")
    elif isinstance(source, Sequence):
        # Sequence of strings should point to files or URLs
        return isinstance(source[0], str)
    elif isinstance(source, Iterable):
        # If we get an iterable but not a sequence, the best we can do
        # is check if we have a known non-distributed object
        if isinstance(source, (pd.DataFrame, pd.Series, np.ndarray)):
            return False

    # Per default, allow distributed loading.
    return True


def _detect_distributed(source: Data) -> bool:
    """Returns True if we should try to use distributed data loading"""
    from xgboost_ray.data_sources.modin import Modin

    if not _can_load_distributed(source):
        return False
    if Modin.is_data_type(source):
        return True
    if isinstance(source, RayDataset):
        return True
    if (
        isinstance(source, Iterable)
        and not isinstance(source, str)
        and not (isinstance(source, Sequence) and isinstance(source[0], str))
    ):
        # This is an iterable but not a Sequence of strings, and not a
        # pandas dataframe, series, or numpy array.
        # Detect False per default, can be overridden by passing
        # `distributed=True` to the RayDMatrix object.
        return False

    # Otherwise, assume distributed loading is possible
    return True



class _RayPoolLoader:

    def __init__(
        self,
        data: Data,
        label: Optional[Data] = None,
        cat_features = None,
        text_features = None,
        embedding_features = None,
        column_description=None,
        pairs=None,
        delimiter='\t',
        has_header=False,
        weight: Optional[Data] = None,
        group_id=None,
        group_weight=None,
        subgroup_id=None,
        pairs_weight=None,
        baseline=None,
        timestamp=None,
        feature_names: Optional[List[str]] = None,
        log_cout=sys.stdout,
        log_cerr=sys.stderr,
        filetype = None,
        ignore = None,
        **kwargs,
    ):
        self.data = data
        self.label = label
        self.cat_features = cat_features
        self.text_features = text_features
        self.embedding_features = embedding_features
        self.column_description = column_description
        self.pairs = pairs
        self.delimiter = delimiter
        self.has_header = has_header
        self.weight = weight
        self.group_id = group_id
        self.group_weight = group_weight
        self.subgroup_id = subgroup_id
        self.pairs_weight = pairs_weight
        self.baseline = baseline
        self.timestamp = timestamp
        self.feature_names = feature_names
        self.log_cout = log_cout
        self.log_cerr = log_cerr
        self.filetype = filetype
        self.ignore = ignore
        self.kwargs = kwargs

        self.data_source = None
        self.actor_shards = None
        self._cached_n = None

        check = None
        if isinstance(data, str):
            check = data
        elif isinstance(data, Sequence) and isinstance(data[0], str):
            check = data[0]

        if check is not None:
            if not self.filetype:
                # Try to guess filetype
                for data_source in data_sources:
                    self.filetype = data_source.get_filetype(check)
                    if self.filetype:
                        break
                if not self.filetype:
                    raise ValueError(
                        f"File or stream ({check}) specified as data source, "
                        "but filetype could not be detected. "
                        "\nFIX THIS by passing "
                        "the `filetype` parameter to the RayDMatrix. Use the "
                        "`RayFileType` enum for this."
                    )
                
    def get_data_source(self) -> Type[DataSource]:
        raise NotImplementedError

    def assert_enough_shards_for_actors(self, num_actors: int):
        """Assert that we have enough shards to split across actors."""
        # Pass per default
        pass

    def update_matrix_properties(self, matrix: "xgb.DMatrix"):
        data_source = self.get_data_source()
        data_source.update_feature_names(matrix, self.feature_names)

    def assign_shards_to_actors(self, actors: Sequence[ActorHandle]) -> bool:
        """Assign data shards to actors.

        Returns True if shards were assigned to actors. In that case, the
        sharding mode should be adjusted to ``RayShardingMode.FIXED``.
        Returns False otherwise.
        """
        return False

    def _split_dataframe(
        self, local_data: pd.DataFrame, data_source: Type[DataSource]
    ) -> Tuple[
        pd.DataFrame,
        Optional[pd.Series],
        Optional[pd.Series],
        Optional[pd.Series],
        Optional[pd.Series],
        Optional[pd.Series],
    ]:
        
        exclude_cols = set()
        
        label, exclude = data_source.get_column(local_data, self.label)
        if exclude:
            exclude_cols.add(exclude)

        weight, exclude = data_source.get_column(local_data, self.weight)
        if exclude:
            exclude_cols.add(exclude)
    

        x = local_data
        if exclude_cols:
            x = x[[col for col in x.columns if col not in exclude_cols]]

        return (
            x,
            label,
            weight,
        )
    
    def load_data(self):
        raise NotImplementedError
        

class _CentralRayPoolLoader(_RayPoolLoader):

    def get_data_source(self):
        if self.data_source:
            return self.data_source

        data_source = None
        for source in data_sources:
            if not source.supports_central_loading:
                continue

            try:
                if source.is_data_type(self.data, self.filetype):
                    data_source = source
                    break
            except Exception as exc:
                # If checking the data throws an exception, the data source
                # is not available.
                logger.warning(
                    f"Checking data source {source.__name__} failed "
                    f"with exception: {exc}"
                )
                continue

        if not data_source:
            raise ValueError(
                "Unknown data source type: {} with FileType: {}."
                "\nFIX THIS by passing a supported data type. Supported "
                "data types include pandas.DataFrame, pandas.Series, "
                "np.ndarray, and CSV/Parquet file paths. If you specify a "
                "file, path, consider passing the `filetype` argument to "
                "specify the type of the source. Use the `RayFileType` "
                "enum for that. If using Modin, Dask, or Petastorm, "
                "make sure the library is installed.".format(
                    type(self.data), self.filetype
                )
            )

        if (
            self.label is not None
            and not isinstance(self.label, str)
            and not type(self.data) != type(self.label)  # noqa: E721
        ):  # noqa: E721:
            # Label is an object of a different type than the main data.
            # We have to make sure they are compatible
            if not data_source.is_data_type(self.label):
                raise ValueError(
                    "The passed `data` and `label` types are not compatible."
                    "\nFIX THIS by passing the same types to the "
                    "`RayDMatrix` - e.g. a `pandas.DataFrame` as `data` "
                    "and `label`. The `label` can always be a string. Got "
                    "{} for the main data and {} for the label.".format(
                        type(self.data), type(self.label)
                    )
                )

        self.data_source = data_source
        self._cached_n = data_source.get_n(self.data)
        return self.data_source
    
    def load_data(self, num_actors, sharding, rank = None):
        if not ray.is_initialized():
            ray.init()

        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]

        data_source = self.get_data_source()

        max_num_shards = self._cached_n or data_source.get_n(self.data)
        if num_actors > max_num_shards and data_source.needs_partitions:
            raise RuntimeError(
                f"Trying to shard data for {num_actors} actors, but the "
                f"maximum number of shards (i.e. the number of data rows) "
                f"is {max_num_shards}. Consider using fewer actors."
            )

        # We're doing central data loading here, so we don't pass any indices,
        # yet. Instead, we'll be selecting the rows below.
        local_df = data_source.load_data(
            self.data, ignore=self.ignore, indices=None, **self.kwargs
        )
        x, y, w = self._split_dataframe(
            local_df, data_source=data_source
        )

        if isinstance(x, list):
            n = sum(len(a) for a in x)
        else:
            n = len(x)

        refs = {}
        for i in range(num_actors):
            # Here we actually want to split the data.
            indices = _get_sharding_indices(sharding, i, num_actors, n)
            actor_refs = {
                "data": ray.put(x.iloc[indices]),
                "label": ray.put(y.iloc[indices] if y is not None else None),
                "weight": ray.put(w.iloc[indices] if w is not None else None),
            }
            refs[i] = actor_refs

        return refs, n
    

class _DistributedRayPoolLoader(_RayPoolLoader):

    def get_data_source(self):
        if self.data_source:
            return self.data_source

        invalid_data = False
        if isinstance(self.data, str):
            if self.filetype == RayFileType.PETASTORM:
                self.data = [self.data]
            elif os.path.isdir(self.data):
                if self.filetype == RayFileType.PARQUET:
                    self.data = sorted(glob.glob(f"{self.data}/**/*.parquet"))
                elif self.filetype == RayFileType.CSV:
                    self.data = sorted(glob.glob(f"{self.data}/**/*.csv"))
                else:
                    invalid_data = True
            elif os.path.exists(self.data):
                self.data = [self.data]
            else:
                invalid_data = True

        # Todo (krfricke): It would be good to have a more general way to
        # check for compatibility here. Combine with test below?
        if (
            not (
                isinstance(self.data, (Iterable, RayDataset))
                or hasattr(self.data, "__partitioned__")
            )
            or invalid_data
        ):
            raise ValueError(
                f"Distributed data loading only works with already "
                f"distributed datasets. These should be specified through a "
                f"list of locations (or a single string). "
                f"Got: {type(self.data)}."
                f"\nFIX THIS by passing a list of files (e.g. on S3) to the "
                f"RayDMatrix."
            )

        if self.label is not None and not isinstance(self.label, str):
            raise ValueError(
                f"Invalid `label` value for distributed datasets: "
                f"{self.label}. Only strings are supported. "
                f"\nFIX THIS by passing a string indicating the label "
                f"column of the dataset as the `label` argument."
            )

        data_source = None
        for source in data_sources:
            if not source.supports_distributed_loading:
                continue

            try:
                if source.is_data_type(self.data, self.filetype):
                    data_source = source
                    break
            except Exception as exc:
                # If checking the data throws an exception, the data source
                # is not available.
                logger.warning(
                    f"Checking data source {source.__name__} failed "
                    f"with exception: {exc}"
                )
                continue

        if not data_source:
            raise ValueError(
                f"Invalid data source type: {type(self.data)} "
                f"with FileType: {self.filetype} for a distributed dataset."
                "\nFIX THIS by passing a supported data type. Supported "
                "data types for distributed datasets are a list of "
                "CSV or Parquet sources. If using "
                "Modin, Dask, or Petastorm, make sure the library is "
                "installed."
            )

        self.data_source = data_source
        self._cached_n = data_source.get_n(self.data)
        return self.data_source
    
    def assert_enough_shards_for_actors(self, num_actors):
        data_source = self.get_data_source()

        # Ray Datasets will be automatically split to match the number
        # of actors.
        if isinstance(data_source, RayDataset):
            return

        max_num_shards = self._cached_n or data_source.get_n(self.data)
        if num_actors > max_num_shards and data_source.needs_partitions:
            raise RuntimeError(
                f"Trying to shard data for {num_actors} actors, but the "
                f"maximum number of shards is {max_num_shards}. If you "
                f"want to shard the dataset by rows, consider "
                f"centralized loading by passing `distributed=False` to "
                f"the `RayDMatrix`. Otherwise consider using fewer actors "
                f"or re-partitioning your data."
            )
    
    def assign_shards_to_actors(self, actors):
        if not isinstance(self.label, str):
            # Currently we only support fixed data sharding for datasets
            # that contain both the label and the data.
            return False

        if self.actor_shards:
            # Only assign once
            return True

        data_source = self.get_data_source()
        data, actor_shards = data_source.get_actor_shards(self.data, actors)
        if not actor_shards:
            return False

        self.data = data
        self.actor_shards = actor_shards
        return True
    
    def load_data(self, num_actors, sharding, rank = None):
        if rank is None or not ray.is_initialized:
            raise ValueError(
                "Distributed loading should be done by the actors, not by the"
                "driver program. "
                "\nFIX THIS by refraining from calling `RayDMatrix.load()` "
                "manually for distributed datasets. Hint: You can check if "
                "`RayDMatrix.distributed` is set to True or False."
            )

        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]

        data_source = self.get_data_source()

        if self.actor_shards:
            if rank is None:
                raise RuntimeError(
                    "Distributed loading requires a rank to be passed, " "got None"
                )
            rank_shards = self.actor_shards[rank]
            local_df = data_source.load_data(
                self.data, indices=rank_shards, ignore=self.ignore, **self.kwargs
            )
            x, y, w = self._split_dataframe(
                local_df, data_source=data_source
            )

            if isinstance(x, list):
                n = sum(len(a) for a in x)
            else:
                n = len(x)
        else:
            n = self._cached_n or data_source.get_n(self.data)
            indices = _get_sharding_indices(sharding, rank, num_actors, n)

            if not indices:
                x, y, w = (
                    None,
                    None,
                    None,
                )
                n = 0
            else:
                local_df = data_source.load_data(
                    self.data, ignore=self.ignore, indices=indices, **self.kwargs
                )
                x, y, w = self._split_dataframe(
                    local_df, data_source=data_source
                )

                if isinstance(x, list):
                    n = sum(len(a) for a in x)
                else:
                    n = len(x)

        refs = {
            rank: {
                "data": ray.put(x),
                "label": ray.put(y),
                "weight": ray.put(w),
            }
        }

        return refs, n
            

@PublicAPI
class RayPool:

    """ Catboost on a Ray Pool Class """

    def __init__(
        self,
        data: Data,
        label: Optional[Data] = None,
        cat_features = None,
        text_features = None,
        embedding_features = None,
        column_description=None,
        pairs=None,
        delimiter='\t',
        has_header=False,
        weight: Optional[Data] = None,
        group_id=None,
        group_weight=None,
        subgroup_id=None,
        pairs_weight=None,
        baseline=None,
        timestamp=None,
        feature_names: Optional[List[str]] = None,
        log_cout=sys.stdout,
        log_cerr=sys.stderr,
        num_actors: Optional[int] = None,
        filetype: Optional[RayFileType] = None,
        ignore: Optional[List[str]] = None,
        distributed: Optional[bool] = None,
        sharding: RayShardingMode = RayShardingMode.INTERLEAVED,
        lazy: bool = False,
        **kwargs,
    ):
        
        if group_weight is not None:
            raise ValueError(
                "per-group weight is not implemented."
            )
        
        self._uid = uuid.uuid4().int
        self.feature_names = feature_names
        self.num_actors = num_actors
        self.sharding = sharding

        if distributed is None:
            distributed = _detect_distributed(data)
        else:
            if distributed and not _can_load_distributed(data):
                raise ValueError(
                    f"You passed `distributed=True` to the `RayDMatrix` but "
                    f"the specified data source of type {type(data)} cannot "
                    f"be loaded in a distributed fashion. "
                    f"\nFIX THIS by passing a list of sources (e.g. parquet "
                    f"files stored in a network location) instead."
                )
            
        self.distributed = distributed

        if self.distributed:
            self.loader = _DistributedRayPoolLoader(
                data=data,
                label=label,
                cat_features=cat_features,
                text_features=text_features,
                embedding_features=embedding_features,
                column_description=column_description,
                pairs=pairs,
                delimiter=delimiter,
                has_header=has_header,
                weight=weight,
                group_id=group_id,
                group_weight=group_weight,
                subgroup_id=subgroup_id,
                pairs_weight=pairs_weight,
                baseline=baseline,
                timestamp=timestamp,
                feature_names=feature_names,
                log_cout=log_cout,
                log_cerr=log_cerr,
                filetype=filetype,
                ignore=ignore,
                **kwargs,
            )
        else:
            self.loader = _CentralRayPoolLoader(
                data=data,
                label=label,
                cat_features=cat_features,
                text_features=text_features,
                embedding_features=embedding_features,
                column_description=column_description,
                pairs=pairs,
                delimiter=delimiter,
                has_header=has_header,
                weight=weight,
                group_id=group_id,
                group_weight=group_weight,
                subgroup_id=subgroup_id,
                pairs_weight=pairs_weight,
                baseline=baseline,
                timestamp=timestamp,
                feature_names=feature_names,
                log_cout=log_cout,
                log_cerr=log_cerr,
                filetype=filetype,
                ignore=ignore,
                **kwargs,
            )

        self.refs: Dict[int, Dict[str, ray.ObjectRef]] = {}
        self.n = None

        self.loaded = False

        if not distributed and num_actors is not None and not lazy:
            self.load_data(num_actors)


    @property
    def has_label(self):
        return self.loader.label is not None

    def assign_shards_to_actors(self, actors: Sequence[ActorHandle]) -> bool:
        success = self.loader.assign_shards_to_actors(actors)
        if success:
            self.sharding = RayShardingMode.FIXED
        return success

    def assert_enough_shards_for_actors(self, num_actors: int):
        self.loader.assert_enough_shards_for_actors(num_actors=num_actors)

    def load_data(self, num_actors: Optional[int] = None, rank: Optional[int] = None):
        """Load data, putting it into the Ray object store.

        If a rank is given, only data for this rank is loaded (for
        distributed data sources only).
        """
        if not self.loaded:
            if num_actors is not None:
                if self.num_actors is not None and num_actors != self.num_actors:
                    raise ValueError(
                        f"The `RayDMatrix` was initialized or `load_data()`"
                        f"has been called with a different numbers of"
                        f"`actors`. Existing value: {self.num_actors}. "
                        f"Current value: {num_actors}."
                        f"\nFIX THIS by not instantiating the matrix with "
                        f"`num_actors` and making sure calls to `load_data()` "
                        f"or `get_data()` use the same numbers of actors "
                        f"at each call."
                    )
                self.num_actors = num_actors
            if self.num_actors is None:
                raise ValueError(
                    "Trying to load data for `RayDMatrix` object, but "
                    "`num_actors` is not set."
                    "\nFIX THIS by passing `num_actors` on instantiation "
                    "of the `RayDMatrix` or when calling `load_data()`."
                )
            refs, self.n = self.loader.load_data(
                self.num_actors, self.sharding, rank=rank
            )
            self.refs.update(refs)
            self.loaded = True

    def get_data(
        self, rank: int, num_actors: Optional[int] = None
    ) -> Dict[str, Union[None, pd.DataFrame, List[Optional[pd.DataFrame]]]]:
        """Get data, i.e. return dataframe for a specific actor.

        This method is called from an actor, given its rank and the
        total number of actors. If the data is not yet loaded, loading
        is triggered.
        """
        self.load_data(num_actors=num_actors, rank=rank)

        refs = self.refs[rank]
        ray.get(list(refs.values()))

        data = {k: ray.get(v) for k, v in refs.items()}

        return data

    def unload_data(self):
        """Delete object references to clear object store"""
        for rank in list(self.refs.keys()):
            for name in list(self.refs[rank].keys()):
                del self.refs[rank][name]
        self.loaded = False

    def update_matrix_properties(self, matrix: "xgb.DMatrix"):
        self.loader.update_matrix_properties(matrix)

    def __hash__(self):
        return self._uid

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

