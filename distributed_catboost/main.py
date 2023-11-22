import gc
import logging
import os
import threading
import multiprocessing
import time
import warnings
import platform
from copy import deepcopy
from dataclasses import dataclass
import struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import socket
from threading import Thread

import ray
import numpy as np
import pandas as pd
from distributed_catboost.catboost import catboost
from distributed_catboost.matrix import RayPool
from xgboost_ray import RayDMatrix
from xgboost_ray.main import (
    DEFAULT_PG,
    ENV,
    LEGACY_MATRIX,
    ActorHandle,
    Event,
    PlacementGroup,
    Queue,
    RayActorError,
    RayDeviceQuantileDMatrix,
)
from xgboost_ray.main import RayParams
from xgboost_ray.main import (
    RayTaskError,
    RayXGBoostActor,
    RayXGBoostActorAvailable,
    RayXGBoostTrainingError,
    _assert_ray_support,
)
from xgboost_ray.main import _autodetect_resources as _autodetect_resources_base
from xgboost_ray.main import (
    _Checkpoint,
    _create_communication_processes,
    _create_placement_group,
    _handle_queue,
    _is_client_connected,
    _maybe_print_legacy_warning,
    _PrepareActorTask,
    _ray_get_actor_cpus,
    _set_omp_num_threads,
    _shutdown,
    _TrainingState,
    _trigger_data_load,
    combine_data,
    concat_dataframes,
    force_on_current_node,
    get_current_placement_group,
    is_session_enabled,
    pickle,
)
from xgboost_ray.session import put_queue, get_rabit_rank, init_session, set_session_queue

from packaging.version import Version
from ray.util.annotations import PublicAPI, DeveloperAPI
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ray.util import get_node_ip_address, placement_group
from xgboost_ray.xgb import xgboost as xgb
from xgboost_ray.callback import DistributedCallback, DistributedCallbackContainer
from xgboost_ray.compat import LEGACY_CALLBACK, TrainingCallback

try:
    from xgboost.collective import CommunicatorContext

    rabit = None
    HAS_COLLECTIVE = True
except ImportError:
    from xgboost import rabit  # noqa

    CommunicatorContext = None
    HAS_COLLECTIVE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RayXGBoostTrainingError(RuntimeError):
    """Raised from RayXGBoostActor.train() when the local xgb.train function
    did not complete."""

    pass


class RayXGBoostTrainingStopped(RuntimeError):
    """Raised from RayXGBoostActor.train() when training was deliberately
    stopped."""

    pass


class RayXGBoostActorAvailable(RuntimeError):
    """Raise from `_update_scheduled_actor_states()` when new actors become
    available in elastic training"""

    pass

class ExSocket(object):
    """
    Extension of socket to handle recv and send of special data
    """

    def __init__(self, sock):
        self.sock = sock

    def recvall(self, nbytes):
        res = []
        nread = 0
        while nread < nbytes:
            chunk = self.sock.recv(min(nbytes - nread, 1024))
            nread += len(chunk)
            res.append(chunk)
        return b"".join(res)

    def recvint(self):
        return struct.unpack("@i", self.recvall(4))[0]

    def sendint(self, n):
        self.sock.sendall(struct.pack("@i", n))

    def sendstr(self, s):
        self.sendint(len(s))
        self.sock.sendall(s.encode())

    def recvstr(self):
        slen = self.recvint()
        return self.recvall(slen).decode()


# magic number used to verify existence of data
kMagic = 0xFF99


def get_some_ip(host):
    return socket.getaddrinfo(host, None)[0][4][0]


def get_host_ip(hostIP=None):
    if hostIP is None or hostIP == "auto":
        hostIP = "ip"

    if hostIP == "dns":
        hostIP = socket.getfqdn()
    elif hostIP == "ip":
        from socket import gaierror

        try:
            hostIP = socket.gethostbyname(socket.getfqdn())
        except gaierror:
            logging.debug(
                "gethostbyname(socket.getfqdn()) failed... trying on hostname()"
            )
            hostIP = socket.gethostbyname(socket.gethostname())
        if hostIP.startswith("127."):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # doesn't have to be reachable
            s.connect(("10.255.255.255", 1))
            hostIP = s.getsockname()[0]
    return hostIP


def get_family(addr):
    return socket.getaddrinfo(addr, None)[0][0]


class SlaveEntry(object):
    def __init__(self, sock, s_addr):
        slave = ExSocket(sock)
        self.sock = slave
        self.host = get_some_ip(s_addr[0])
        magic = slave.recvint()
        assert magic == kMagic, "invalid magic number=%d from %s" % (magic, self.host)
        slave.sendint(kMagic)
        self.rank = slave.recvint()
        self.world_size = slave.recvint()
        self.jobid = slave.recvstr()
        self.cmd = slave.recvstr()
        self.wait_accept = 0
        self.port = None

    def decide_rank(self, job_map):
        if self.rank >= 0:
            return self.rank
        if self.jobid != "NULL" and self.jobid in job_map:
            return job_map[self.jobid]
        return -1

    def assign_rank(self, rank, wait_conn, tree_map, parent_map, ring_map):
        self.rank = rank
        nnset = set(tree_map[rank])
        rprev, rnext = ring_map[rank]
        self.sock.sendint(rank)
        # send parent rank
        self.sock.sendint(parent_map[rank])
        # send world size
        self.sock.sendint(len(tree_map))
        self.sock.sendint(len(nnset))
        # send the rprev and next link
        for r in nnset:
            self.sock.sendint(r)
        # send prev link
        if rprev not in (-1, rank):
            nnset.add(rprev)
            self.sock.sendint(rprev)
        else:
            self.sock.sendint(-1)
        # send next link
        if rnext not in (-1, rank):
            nnset.add(rnext)
            self.sock.sendint(rnext)
        else:
            self.sock.sendint(-1)
        while True:
            ngood = self.sock.recvint()
            goodset = set([])
            for _ in range(ngood):
                goodset.add(self.sock.recvint())
            assert goodset.issubset(nnset)
            badset = nnset - goodset
            conset = []
            for r in badset:
                if r in wait_conn:
                    conset.append(r)
            self.sock.sendint(len(conset))
            self.sock.sendint(len(badset) - len(conset))
            for r in conset:
                self.sock.sendstr(wait_conn[r].host)
                self.sock.sendint(wait_conn[r].port)
                self.sock.sendint(r)
            nerr = self.sock.recvint()
            if nerr != 0:
                continue
            self.port = self.sock.recvint()
            rmset = []
            # all connection was successuly setup
            for r in conset:
                wait_conn[r].wait_accept -= 1
                if wait_conn[r].wait_accept == 0:
                    rmset.append(r)
            for r in rmset:
                wait_conn.pop(r, None)
            self.wait_accept = len(badset) - len(conset)
            return rmset


class RabitTracker(object):
    """
    tracker for rabit
    """

    def __init__(self, hostIP, nslave, port=9091, port_end=9999):
        sock = socket.socket(get_family(hostIP), socket.SOCK_STREAM)
        for _port in range(port, port_end):
            try:
                sock.bind((hostIP, _port))
                self.port = _port
                break
            except socket.error as e:
                if e.errno in [98, 48]:
                    continue
                raise
        sock.listen(256)
        self.sock = sock
        self.hostIP = hostIP
        self.thread = None
        self.start_time = None
        self.end_time = None
        self.nslave = nslave
        logging.info("start listen on %s:%d", hostIP, self.port)

    def __del__(self):
        self.sock.close()

    @staticmethod
    def get_neighbor(rank, nslave):
        rank = rank + 1
        ret = []
        if rank > 1:
            ret.append(rank // 2 - 1)
        if rank * 2 - 1 < nslave:
            ret.append(rank * 2 - 1)
        if rank * 2 < nslave:
            ret.append(rank * 2)
        return ret

    def slave_envs(self):
        """
        get enviroment variables for slaves
        can be passed in as args or envs
        """
        return {"DMLC_TRACKER_URI": self.hostIP, "DMLC_TRACKER_PORT": self.port}

    def get_tree(self, nslave):
        tree_map = {}
        parent_map = {}
        for r in range(nslave):
            tree_map[r] = self.get_neighbor(r, nslave)
            parent_map[r] = (r + 1) // 2 - 1
        return tree_map, parent_map

    def find_share_ring(self, tree_map, parent_map, r):
        """
        get a ring structure that tends to share nodes with the tree
        return a list starting from r
        """
        nset = set(tree_map[r])
        cset = nset - set([parent_map[r]])
        if not cset:
            return [r]
        rlst = [r]
        cnt = 0
        for v in cset:
            vlst = self.find_share_ring(tree_map, parent_map, v)
            cnt += 1
            if cnt == len(cset):
                vlst.reverse()
            rlst += vlst
        return rlst

    def get_ring(self, tree_map, parent_map):
        """
        get a ring connection used to recover local data
        """
        assert parent_map[0] == -1
        rlst = self.find_share_ring(tree_map, parent_map, 0)
        assert len(rlst) == len(tree_map)
        ring_map = {}
        nslave = len(tree_map)
        for r in range(nslave):
            rprev = (r + nslave - 1) % nslave
            rnext = (r + 1) % nslave
            ring_map[rlst[r]] = (rlst[rprev], rlst[rnext])
        return ring_map

    def get_link_map(self, nslave):
        """
        get the link map, this is a bit hacky, call for better algorithm
        to place similar nodes together
        """
        tree_map, parent_map = self.get_tree(nslave)
        ring_map = self.get_ring(tree_map, parent_map)
        rmap = {0: 0}
        k = 0
        for i in range(nslave - 1):
            k = ring_map[k][1]
            rmap[k] = i + 1

        ring_map_ = {}
        tree_map_ = {}
        parent_map_ = {}
        for k, v in ring_map.items():
            ring_map_[rmap[k]] = (rmap[v[0]], rmap[v[1]])
        for k, v in tree_map.items():
            tree_map_[rmap[k]] = [rmap[x] for x in v]
        for k, v in parent_map.items():
            if k != 0:
                parent_map_[rmap[k]] = rmap[v]
            else:
                parent_map_[rmap[k]] = -1
        return tree_map_, parent_map_, ring_map_

    def accept_slaves(self, nslave):
        # set of nodes that finishs the job
        shutdown = {}
        # set of nodes that is waiting for connections
        wait_conn = {}
        # maps job id to rank
        job_map = {}
        # list of workers that is pending to be assigned rank
        pending = []
        # lazy initialize tree_map
        tree_map = None

        while len(shutdown) != nslave:
            fd, s_addr = self.sock.accept()
            s = SlaveEntry(fd, s_addr)
            if s.cmd == "print":
                msg = s.sock.recvstr()
                print(msg.strip(), flush=True)
                continue
            if s.cmd == "shutdown":
                assert s.rank >= 0 and s.rank not in shutdown
                assert s.rank not in wait_conn
                shutdown[s.rank] = s
                logging.debug("Received %s signal from %d", s.cmd, s.rank)
                continue
            assert s.cmd == "start" or s.cmd == "recover"
            # lazily initialize the slaves
            if tree_map is None:
                assert s.cmd == "start"
                if s.world_size > 0:
                    nslave = s.world_size
                tree_map, parent_map, ring_map = self.get_link_map(nslave)
                # set of nodes that is pending for getting up
                todo_nodes = list(range(nslave))
            else:
                assert s.world_size == -1 or s.world_size == nslave
            if s.cmd == "recover":
                assert s.rank >= 0

            rank = s.decide_rank(job_map)
            # batch assignment of ranks
            if rank == -1:
                assert todo_nodes
                pending.append(s)
                if len(pending) == len(todo_nodes):
                    pending.sort(key=lambda x: x.host)
                    for s in pending:
                        rank = todo_nodes.pop(0)
                        if s.jobid != "NULL":
                            job_map[s.jobid] = rank
                        s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
                        if s.wait_accept > 0:
                            wait_conn[rank] = s
                        logging.debug(
                            "Received %s signal from %s; assign rank %d",
                            s.cmd,
                            s.host,
                            s.rank,
                        )
                if not todo_nodes:
                    logging.info("@tracker All of %d nodes getting started", nslave)
                    self.start_time = time.time()
            else:
                s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
                logging.debug("Received %s signal from %d", s.cmd, s.rank)
                if s.wait_accept > 0:
                    wait_conn[rank] = s
        logging.info("@tracker All nodes finishes job")
        self.end_time = time.time()
        logging.info(
            "@tracker %s secs between node start and job finish",
            str(self.end_time - self.start_time),
        )

    def start(self, nslave):
        def run():
            self.accept_slaves(nslave)

        self.thread = Thread(target=run, args=(), daemon=True)
        self.thread.start()

    def join(self):
        while self.thread.is_alive():
            self.thread.join(100)

    def alive(self):
        return self.thread.is_alive()

class _RabitTrackerCompatMixin:
    """Fallback calls to legacy terminology"""

    def accept_workers(self, n_workers: int):
        return self.accept_slaves(n_workers)

    def worker_envs(self):
        return self.slave_envs()


class _RabitTracker(RabitTracker, _RabitTrackerCompatMixin):
    """
    This method overwrites the xgboost-provided RabitTracker to switch
    from a daemon thread to a multiprocessing Process. This is so that
    we are able to terminate/kill the tracking process at will.
    """

    def start(self, nworker):
        # TODO: refactor RabitTracker to support spawn process creation.
        # In python 3.8, spawn is used as default process creation on macOS.
        # But spawn doesn't work because `run` is not pickleable.
        # For now we force the start method to use fork.
        multiprocessing.set_start_method("fork", force=True)

        def run():
            self.accept_workers(nworker)

        self.thread = multiprocessing.Process(target=run, args=())
        self.thread.start()


def _start_rabit_tracker(num_workers: int):
    """Start Rabit tracker. The workers connect to this tracker to share
    their results.

    The Rabit tracker is the main process that all local workers connect to
    to share their weights. When one or more actors die, we want to
    restart the Rabit tracker, too, for two reasons: First we don't want to
    be potentially stuck with stale connections from old training processes.
    Second, we might restart training with a different number of actors, and
    for that we would have to restart the tracker anyway.

    To do this we start the Tracker in its own subprocess with its own PID.
    We can use this process then to specifically kill/terminate the tracker
    process in `_stop_rabit_tracker` without touching other functionality.
    """
    host = get_node_ip_address()

    env = {"DMLC_NUM_WORKER": num_workers}

    rabit_tracker = _RabitTracker(host, num_workers)

    # Get tracker Host + IP
    env.update(rabit_tracker.worker_envs())
    rabit_tracker.start(num_workers)

    logger.debug(f"Started Rabit tracker process with PID {rabit_tracker.thread.pid}")

    return rabit_tracker.thread, env


def _stop_rabit_tracker(rabit_process: multiprocessing.Process):
    logger.debug(f"Stopping Rabit process with PID {rabit_process.pid}")
    rabit_process.join(timeout=5)
    rabit_process.terminate()


class _RabitContextBase:
    """This context is used by local training actors to connect to the
    Rabit tracker.

    Args:
        actor_id: Unique actor ID
        args: Arguments for Rabit initialisation. These are
            environment variables to configure Rabit clients.
    """

    def __init__(self, actor_id: int, args: dict):
        args["DMLC_TASK_ID"] = "[xgboost.ray]:" + actor_id
        self.args = args


# From xgboost>=1.7.0, rabit is replaced by a collective communicator
if HAS_COLLECTIVE:

    class _RabitContext(_RabitContextBase, CommunicatorContext):
        pass

else:

    class _RabitContext(_RabitContextBase):
        def __init__(self, actor_id: int, args: dict):
            super().__init__(actor_id, args)
            self._list_args = [("%s=%s" % item).encode() for item in self.args.items()]

        def __enter__(self):
            xgb.rabit.init(self._list_args)

        def __exit__(self, *args):
            xgb.rabit.finalize()


# @dataclass
# class RayParams(RayXGBParams):
#     # The RayParams from XGBoost-Ray can also be used, in which
#     # case allow_less_than_two_cpus will just default to False
#     allow_less_than_two_cpus: bool = False

#     __doc__ = RayXGBParams.__doc__.replace(
#         """        elastic_training: If True, training will continue with
#             fewer actors if an actor fails. Default False.""",
#         """        allow_less_than_two_cpus: If True, an exception will not
#             be raised if `cpus_per_actor`. Default False.""",
#     ).replace(
#         """cpus_per_actor: Number of CPUs to be used per Ray actor.""",
#         """cpus_per_actor: Number of CPUs to be used per Ray actor.
#             If smaller than 2, training might be substantially slower
#             because communication work and training work will block
#             each other. This will raise an exception unless
#             `allow_less_than_two_cpus` is True.""",
#     )

#     def get_tune_resources(self):
#         _check_cpus_per_actor_at_least_2(
#             self.cpus_per_actor, getattr(self, "allow_less_than_two_cpus", False)
#         )
#         return super().get_tune_resources()


def _validate_ray_params(ray_params: Union[None, RayParams, dict]) -> RayParams:
    if ray_params is None:
        ray_params = RayParams()
    elif isinstance(ray_params, dict):
        ray_params = RayParams(**ray_params)
    elif not isinstance(ray_params, RayParams):
        raise ValueError(
            f"`ray_params` must be a `RayParams` instance, a dict, or None, "
            f"but it was {type(ray_params)}."
            f"\nFIX THIS preferably by passing a `RayParams` instance as "
            f"the `ray_params` parameter."
        )
    if ray_params.num_actors <= 0:
        raise ValueError(
            "The `num_actors` parameter is set to 0. Please always specify "
            "the number of distributed actors you want to use."
            "\nFIX THIS by passing a `RayParams(num_actors=X)` argument "
            "to your call to lightgbm_ray."
        )
    elif ray_params.num_actors < 2:
        warnings.warn(
            f"`num_actors` in `ray_params` is smaller than 2 "
            f"({ray_params.num_actors}). LightGBM will NOT be distributed!"
        )
    return ray_params


class RayCatboostActor(RayXGBoostActor):

    """ Remote Ray Catboost Actor Class
    """

    def __init__(
        self,
        rank: int,
        num_actors: int,
        queue: Optional[Queue] = None,
        stop_event: Optional[Event] = None,
        checkpoint_frequency: int = 5,
        distributed_callbacks: Optional[List[DistributedCallback]] = None,
    ):
        self.queue = queue
        init_session(rank, self.queue)

        self.rank = rank
        self.num_actors = num_actors

        self.checkpoint_frequency = checkpoint_frequency

        self._data: Dict[RayDMatrix, xgb.DMatrix] = {}
        self._local_n: Dict[RayDMatrix, int] = {}

        self._stop_event = stop_event

        self._distributed_callbacks = DistributedCallbackContainer(
            distributed_callbacks
        )

        self._distributed_callbacks.on_init(self)
        _set_omp_num_threads()
        logger.debug(f"Initialized remote Catboost actor with rank {self.rank}")

    def set_queue(self, queue: Queue):
        self.queue = queue
        set_session_queue(self.queue)

    def set_stop_event(self, stop_event: Event):
        self._stop_event = stop_event

    def _get_stop_event(self):
        return self._stop_event

    def pid(self):
        """Get process PID. Used for checking if still alive"""
        return os.getpid()

    def ip(self):
        """Get node IP address."""
        return get_node_ip_address()

    def _save_checkpoint_callback(self):
        """Send checkpoints to driver"""
        this = self

        class _SaveInternalCheckpointCallback(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if get_rabit_rank() == 0 and epoch % this.checkpoint_frequency == 0:
                    put_queue(_Checkpoint(epoch, pickle.dumps(model)))

            def after_training(self, model):
                if get_rabit_rank() == 0:
                    put_queue(_Checkpoint(-1, pickle.dumps(model)))
                return model

        return _SaveInternalCheckpointCallback()

    def _stop_callback(self):
        """Stop if event is set"""
        this = self
        # Keep track of initial stop event. Since we're training in a thread,
        # the stop event might be overwritten, which should he handled
        # as if the previous stop event was set.
        initial_stop_event = self._stop_event

        class _StopCallback(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                try:
                    if (
                        this._stop_event.is_set()
                        or this._get_stop_event() is not initial_stop_event
                    ):
                        # Returning True stops training
                        return True
                except RayActorError:
                    return True

        return _StopCallback()

    def load_data(self, data: RayDMatrix):
        if data in self._data:
            return

        self._distributed_callbacks.before_data_loading(self, data)

        param = data.get_data(self.rank, self.num_actors)
        if isinstance(param["data"], list):
            self._local_n[data] = sum(len(a) for a in param["data"])
        else:
            self._local_n[data] = len(param["data"])

        # set nthread for dmatrix conversion
        param["nthread"] = int(_ray_get_actor_cpus())
        self._data[data] = param

        self._distributed_callbacks.after_data_loading(self, data)

    def train(
        self,
        rabit_args: List[str],
        return_bst: bool,
        params: Dict[str, Any],
        dtrain: RayDMatrix,
        evals: Tuple[RayDMatrix, str],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        
        self._distributed_callbacks.before_train(self)

        num_threads = _set_omp_num_threads()

        local_params = params.copy()

        if "init_model" in kwargs:
            if isinstance(kwargs["init_model"], bytes):
                # bytearray type gets lost in remote actor call
                kwargs["xgb_model"] = bytearray(kwargs["xgb_model"])

        if "thread_count" not in local_params:
            if num_threads > 0:
                local_params["thread_count"] = num_threads
            else:
                local_params["thread_count"] = _ray_get_actor_cpus()

        if dtrain not in self._data:
            self.load_data(dtrain)

        for deval, _name in evals:
            if deval not in self._data:
                self.load_data(deval)

        evals_result = dict()

        if "callbacks" in kwargs:
            callbacks = kwargs["callbacks"] or []
        else:
            callbacks = []
        callbacks.append(self._save_checkpoint_callback())
        callbacks.append(self._stop_callback())
        kwargs["callbacks"] = callbacks

        result_dict = {}
        error_dict = {}

        def _train():

            try:
                with _RabitContext(str(id(self)), rabit_args):
                    
                    local_pool = _get_pool(dtrain, self._data[dtrain])

                    if not local_pool.get_label().size:
                        raise RuntimeError(
                            "Training data has no label set. Please make sure "
                            "to set the `label` argument when initializing "
                            "`RayPool()` for data you would like "
                            "to train on."
                        )
                    
                    local_evals = []
                    for deval, name in evals:
                        local_evals.append(
                            (_get_pool(deval, self._data[deval]), name)
                        )
                    
                    bst = catboost.train(
                        local_pool,
                        local_params,
                        evals=local_evals,
                        **kwargs,
                    )

                    if LEGACY_CALLBACK:
                        for xgb_callback in kwargs.get("callbacks", []):
                            if isinstance(xgb_callback, TrainingCallback):
                                xgb_callback.after_training(bst)

                    result_dict.update(
                        {
                            "bst": bst,
                            "train_n": self._local_n[dtrain],
                        }
                    )

            except catboost.CatBoostError as e:
                error_dict.update({"exception": e})
                return
            
        thread = threading.Thread(target=_train)
        thread.daemon = True
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0)
            if self._stop_event.is_set():
                raise RayXGBoostTrainingStopped("Training was interrupted.")
            time.sleep(0.1)

        if not result_dict:
            raise_from = error_dict.get("exception", None)
            raise RayXGBoostTrainingError("Training failed.") from raise_from
        
        thread.join()

        if not return_bst:
            result_dict.pop("bst", None)

        return result_dict
    

    def predict(self, model, data, **kwargs):

        if data not in self._data:
            self.load_data(data)
        local_data = _get_pool(data, self._data[data])

        predictions = model.predict(local_data, **kwargs)
        return predictions



def _prepare_pool_params(param: Dict):
    pool_params = {
        "data": concat_dataframes(param["data"]),
        "label": concat_dataframes(param["label"]),
        "weight": concat_dataframes(param["weight"]),
    }


def _get_pool(data: RayDMatrix, param: Dict):

    if isinstance(param, list):
        dm_param = _prepare_pool_params(param)
        param.update(dm_param)

    matrix = catboost.Pool(**param)
    data.update_matrix_properties(matrix)
    return matrix

    



@ray.remote
class _RemoteRayCatboostActor(RayCatboostActor):
    pass



def _create_actor(
    rank: int,
    num_actors: int,
    num_cpus_per_actor: int,
    num_gpus_per_actor: int,
    resources_per_actor: Optional[Dict] = None,
    placement_group: Optional[PlacementGroup] = None,
    queue: Optional[Queue] = None,
    checkpoint_frequency: int = 5,
    distributed_callbacks: Optional[Sequence[DistributedCallback]] = None,
):
    
    actor_cls = _RemoteRayCatboostActor.options(
        num_cpus=num_cpus_per_actor,
        num_gpus=num_gpus_per_actor,
        resources=resources_per_actor,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group or DEFAULT_PG,
            placement_group_capture_child_tasks=True,
        ),
    )

    return actor_cls.remote(
        rank=rank,
        num_actors=num_actors,
        queue=queue,
        checkpoint_frequency=checkpoint_frequency,
        distributed_callbacks=distributed_callbacks,
    )

@DeveloperAPI
class MultiActorTask:
    """Utility class to hold multiple futures.

    The `is_ready()` method will return True once all futures are ready.

    Args:
        pending_futures: List of object references (futures)
            that should be tracked.
    """

    def __init__(self, pending_futures: Optional[List[ray.ObjectRef]] = None):
        self._pending_futures = pending_futures or []
        self._ready_futures = []

    def is_ready(self):
        if not self._pending_futures:
            return True

        ready = True
        while ready:
            ready, not_ready = ray.wait(self._pending_futures, timeout=0)
            if ready:
                for obj in ready:
                    self._pending_futures.remove(obj)
                    self._ready_futures.append(obj)

        return not bool(self._pending_futures)
    

class _PrepareActorTask(MultiActorTask):
    def __init__(
        self,
        actor: ActorHandle,
        queue: Queue,
        stop_event: Event,
        load_data: List[RayDMatrix],
    ):
        futures = []
        futures.append(actor.set_queue.remote(queue))
        futures.append(actor.set_stop_event.remote(stop_event))
        for data in load_data:
            futures.append(actor.load_data.remote(data))

        super(_PrepareActorTask, self).__init__(futures)


def _train(
    params: Dict,
    dtrain: RayDMatrix,
    *args,
    evals=(),
    ray_params: RayParams,
    cpus_per_actor: int,
    gpus_per_actor: int,
    _training_state: _TrainingState,
    **kwargs,
):
    from xgboost_ray.elastic import (
        _get_actor_alive_status,
        _maybe_schedule_new_actors,
        _update_scheduled_actor_states,
    )

    params = deepcopy(params)

    _training_state.restart_training_at = None

    if "thread_count" in params:
        if params["thread_count"] > cpus_per_actor:
            raise ValueError(
                "Specified number of threads greater than number of CPUs. "
                "\nFIX THIS by passing a lower value for the `n_jobs` "
                "parameter or a higher number for `cpus_per_actor`."
            )
    else:
        params["thread_count"] = cpus_per_actor

    if ray_params.verbose:
        maybe_log = logger.info
        params.setdefault("verbosity", 1)
    else:
        maybe_log = logger.debug
        params.setdefault("verbosity", 0)

    def handle_actor_failure(actor_id):
        rank = _training_state.actors.index(actor_id)
        _training_state.failed_actor_ranks.add(rank)
        _training_state.actors[rank] = None

    newly_created = 0

    for i in list(_training_state.failed_actor_ranks):
        if _training_state.actors[i] is not None:
            raise RuntimeError(
                f"Trying to create actor with rank {i}, but it already " f"exists."
            )
        
        actor = _create_actor(
            rank=i,
            num_actors=ray_params.num_actors,
            num_cpus_per_actor=cpus_per_actor,
            num_gpus_per_actor=gpus_per_actor,
            resources_per_actor=ray_params.resources_per_actor,
            placement_group=_training_state.placement_group,
            queue=_training_state.queue,
            checkpoint_frequency=ray_params.checkpoint_frequency,
            distributed_callbacks=ray_params.distributed_callbacks,
        )

        _training_state.actors[i] = actor
        # Remove from this set so it is not created again
        _training_state.failed_actor_ranks.remove(i)
        newly_created += 1

    alive_actors = sum(1 for a in _training_state.actors if a is not None)

    maybe_log(
        f"[RayXGBoost] Created {newly_created} new actors "
        f"({alive_actors} total actors). Waiting until actors "
        f"are ready for training."
    )

    # For distributed datasets (e.g. Modin), this will initialize
    # (and fix) the assignment of data shards to actor ranks
    dtrain.assert_enough_shards_for_actors(num_actors=ray_params.num_actors)
    dtrain.assign_shards_to_actors(_training_state.actors)
    for deval, _ in evals:
        deval.assert_enough_shards_for_actors(num_actors=ray_params.num_actors)
        deval.assign_shards_to_actors(_training_state.actors)

    load_data = [dtrain] + [eval[0] for eval in evals]

    prepare_actor_tasks = [
        _PrepareActorTask(
            actor,
            # Maybe we got a new Queue actor, so send it to all actors.
            queue=_training_state.queue,
            # Maybe we got a new Event actor, so send it to all actors.
            stop_event=_training_state.stop_event,
            # Trigger data loading
            load_data=load_data,
        )
        for actor in _training_state.actors
        if actor is not None
    ]

    start_wait = time.time()
    last_status = start_wait
    try:
        # Construct list before calling any() to force evaluation
        ready_states = [task.is_ready() for task in prepare_actor_tasks]
        while not all(ready_states):
            if time.time() >= last_status + ENV.STATUS_FREQUENCY_S:
                wait_time = time.time() - start_wait
                logger.info(
                    f"Waiting until actors are ready "
                    f"({wait_time:.0f} seconds passed)."
                )
                last_status = time.time()
            time.sleep(0.1)
            ready_states = [task.is_ready() for task in prepare_actor_tasks]

    except Exception as exc:
        _training_state.stop_event.set()
        _get_actor_alive_status(_training_state.actors, handle_actor_failure)
        raise RayActorError from exc

    maybe_log("[RayXGBoost] Starting XGBoost training.")

    rabit_process, rabit_args = _start_rabit_tracker(alive_actors)

    if _training_state.checkpoint.value:
        kwargs["init_model"] = pickle.loads(_training_state.checkpoint.value)
        if _training_state.checkpoint.iteration == -1:
            # -1 means training already finished.
            logger.error(
                "Trying to load continue from checkpoint, but the checkpoint"
                "indicates training already finished. Returning last"
                "checkpointed model instead."
            )
            return kwargs["init_model"], {}, _training_state.additional_results
        
    callback_returns = _training_state.additional_results.get("callback_returns")
    if callback_returns is None:
        callback_returns = [list() for _ in range(len(_training_state.actors))]
        _training_state.additional_results["callback_returns"] = callback_returns

    _training_state.training_started_at = time.time()
    live_actors = [actor for actor in _training_state.actors if actor is not None]
    training_futures = [
        actor.train.remote(
            rabit_args, i == 0, params, dtrain, evals, *args, **kwargs  # return_bst
        )
        for i, actor in enumerate(live_actors)
    ]

    start_wait = time.time()
    last_status = start_wait

    has_queue_been_handled = False
    try:
        not_ready = training_futures
        while not_ready:
            if _training_state.queue:
                has_queue_been_handled = True
                _handle_queue(
                    queue=_training_state.queue,
                    checkpoint=_training_state.checkpoint,
                    callback_returns=callback_returns,
                )

            if ray_params.elastic_training and not ENV.ELASTIC_RESTART_DISABLED:
                _maybe_schedule_new_actors(
                    training_state=_training_state,
                    num_cpus_per_actor=cpus_per_actor,
                    num_gpus_per_actor=gpus_per_actor,
                    resources_per_actor=ray_params.resources_per_actor,
                    ray_params=ray_params,
                    load_data=load_data,
                )

                # This may raise RayXGBoostActorAvailable
                _update_scheduled_actor_states(_training_state)

            if time.time() >= last_status + ENV.STATUS_FREQUENCY_S:
                wait_time = time.time() - start_wait
                logger.info(
                    f"Training in progress "
                    f"({wait_time:.0f} seconds since last restart)."
                )
                last_status = time.time()

            ready, not_ready = ray.wait(
                not_ready, num_returns=len(not_ready), timeout=1
            )
            ray.get(ready)

        # Get items from queue one last time
        if not has_queue_been_handled:
            time.sleep(1)
        if _training_state.queue:
            _handle_queue(
                queue=_training_state.queue,
                checkpoint=_training_state.checkpoint,
                callback_returns=callback_returns,
            )
    
    except Exception as exc:
        logger.debug(f"Caught exception in training loop: {exc}")

        # Stop all other actors from training
        _training_state.stop_event.set()

        # Check which actors are still alive
        _get_actor_alive_status(_training_state.actors, handle_actor_failure)

        # Todo: Try to fetch newer checkpoint, store in `_checkpoint`
        # Shut down rabit
        _stop_rabit_tracker(rabit_process)

        raise RayActorError from exc
    
    _stop_rabit_tracker(rabit_process)

    # Get all results from all actors.
    all_results: List[Dict[str, Any]] = ray.get(training_futures)

    # All results should be the same because of Rabit tracking. But only
    # the first one actually returns its bst object.
    bst = all_results[0]["bst"]
    evals_result = all_results[0]["evals_result"]

    if callback_returns:
        _training_state.additional_results["callback_returns"] = callback_returns

    total_n = sum(res["train_n"] or 0 for res in all_results)

    _training_state.additional_results["total_n"] = total_n

    return bst, evals_result, _training_state.additional_results



@PublicAPI
def train(
    params: Dict,
    dtrain: RayDMatrix,
    num_boost_round: int = 10,
    *args,
    evals: Union[List[Tuple[RayDMatrix, str]], Tuple[RayDMatrix, str]] = (),
    evals_result: Optional[Dict] = None,
    additional_results: Optional[Dict] = None,
    ray_params: Union[None, RayParams, Dict] = None,
    _remote: Optional[bool] = None,
    **kwargs,
):
    
    os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

    # if platform.system() == "Windows":
    #     raise RuntimeError(
    #         "xgboost-ray training currently does not support " "Windows."
    #     )
    
    if catboost is None:
        raise ImportError(
            "The catboost package is not installed. Distributed_catboost "
            'cannot work without it. Please install catboost using "pip install distributed_catboost"'
        )
    
    if _remote:
        # Run this function as a remote function to support Ray client mode.
        @ray.remote(num_cpus=0)
        def _wrapped(*args, **kwargs):
            _evals_result = {}
            _additional_results = {}
            bst = train(
                *args,
                num_boost_round=num_boost_round,
                evals_result=_evals_result,
                additional_results=_additional_results,
                **kwargs,
            )
            return bst, _evals_result, _additional_results

        # Make sure that train is called on the server node.
        _wrapped = force_on_current_node(_wrapped)

        bst, train_evals_result, train_additional_results = ray.get(
            _wrapped.remote(
                params,
                dtrain,
                *args,
                evals=evals,
                ray_params=ray_params,
                _remote=False,
                **kwargs,
            )
        )
        if isinstance(evals_result, dict):
            evals_result.update(train_evals_result)
        if isinstance(additional_results, dict):
            additional_results.update(train_additional_results)
        return bst
    

    start_time = time.time()
    ray_params = _validate_ray_params(ray_params)
    params = params.copy()

    max_actor_restarts = (
        ray_params.max_actor_restarts
        if ray_params.max_actor_restarts >= 0
        else float("inf")
    )
    _assert_ray_support()

    if not isinstance(dtrain, RayDMatrix):
        raise ValueError(
            "The `dtrain` argument passed to `train()` is not a RayDMatrix, "
            "but of type {}. "
            "\nFIX THIS by instantiating a RayDMatrix first: "
            "`dtrain = RayDMatrix(data=data, label=label)`.".format(type(dtrain))
        )
    
    # LightGBM currently does not support elastic training.
    if ray_params.elastic_training:
        raise ValueError(
            "Elastic Training cannot be used with Catboost at this time. "
            "Please disable elastic_training in `ray_params` "
            "in order to use LightGBM-Ray."
        )
    
    if get_current_placement_group():
        cpus_per_actor = ray_params.cpus_per_actor
        gpus_per_actor = max(0, ray_params.gpus_per_actor)
    else:
        cpus_per_actor, gpus_per_actor = _autodetect_resources(
            ray_params=ray_params,
        )

    if cpus_per_actor == 0 and gpus_per_actor == 0:
        raise ValueError(
            "cpus_per_actor and gpus_per_actor both cannot be "
            "0. Are you sure your cluster has CPUs available?"
        )
    
    if ray_params.elastic_training and ray_params.max_failed_actors == 0:
        raise ValueError(
            "Elastic training enabled but the maximum number of failed "
            "actors is set to 0. This means that elastic training is "
            "effectively disabled. Please set `RayParams.max_failed_actors` "
            "to something larger than 0 to enable elastic training."
        )

    if ray_params.elastic_training and ray_params.max_actor_restarts == 0:
        raise ValueError(
            "Elastic training enabled but the maximum number of actor "
            "restarts is set to 0. This means that elastic training is "
            "effectively disabled. Please set `RayParams.max_actor_restarts` "
            "to something larger than 0 to enable elastic training."
        )

    if not dtrain.has_label:
        raise ValueError(
            "Training data has no label set. Please make sure to set "
            "the `label` argument when initializing `RayDMatrix()` "
            "for data you would like to train on."
        )
    
    if not dtrain.loaded and not dtrain.distributed:
        dtrain.load_data(ray_params.num_actors)

    
    if evals:
        for (deval, _name) in evals:
            if not isinstance(deval, RayDMatrix):
                raise ValueError(
                    "Evaluation data must be a `RayDMatrix`, got " f"{type(deval)}."
                )
            if not deval.has_label:
                raise ValueError(
                    "Evaluation data has no label set. Please make sure to set"
                    " the `label` argument when initializing `RayDMatrix()` "
                    "for data you would like to evaluate on."
                )
            if not deval.loaded and not deval.distributed:
                deval.load_data(ray_params.num_actors)

    bst = None
    train_evals_result = {}
    train_additional_results = {}

    tries = 0
    checkpoint = _Checkpoint()  # Keep track of latest checkpoint
    current_results = {}  # Keep track of additional results
    actors = [None] * ray_params.num_actors  # All active actors
    pending_actors = {}

    queue, stop_event = _create_communication_processes()

    placement_strategy = None
    if not ray_params.elastic_training:
        if get_current_placement_group():
            # Tune is using placement groups, so the strategy has already
            # been set. Don't create an additional placement_group here.
            placement_strategy = None
        elif bool(ENV.USE_SPREAD_STRATEGY):
            placement_strategy = "SPREAD"

    if placement_strategy is not None:
        pg = _create_placement_group(
            cpus_per_actor,
            gpus_per_actor,
            ray_params.resources_per_actor,
            ray_params.num_actors,
            placement_strategy,
        )
    else:
        pg = None

    start_actor_ranks = set(range(ray_params.num_actors))

    total_training_time = 0.0
    boost_rounds_left = num_boost_round
    last_checkpoint_value = checkpoint.value

    while tries <= max_actor_restarts:
        # Only update number of iterations if the checkpoint changed
        # If it didn't change, we already subtracted the iterations.
        if checkpoint.iteration >= 0 and checkpoint.value != last_checkpoint_value:
            boost_rounds_left -= checkpoint.iteration + 1

        last_checkpoint_value = checkpoint.value

        logger.debug(f"Boost rounds left: {boost_rounds_left}")

        training_state = _TrainingState(
            actors=actors,
            queue=queue,
            stop_event=stop_event,
            checkpoint=checkpoint,
            additional_results=current_results,
            training_started_at=0.0,
            placement_group=pg,
            failed_actor_ranks=start_actor_ranks,
            pending_actors=pending_actors,
        )

        try:
            bst, train_evals_result, train_additional_results = _train(
                params,
                dtrain,
                boost_rounds_left,
                *args,
                evals=evals,
                ray_params=ray_params,
                cpus_per_actor=cpus_per_actor,
                gpus_per_actor=gpus_per_actor,
                _training_state=training_state,
                **kwargs,
            )
            if training_state.training_started_at > 0.0:
                total_training_time += time.time() - training_state.training_started_at
            break
        except (RayActorError, RayTaskError) as exc:
            if training_state.training_started_at > 0.0:
                total_training_time += time.time() - training_state.training_started_at
            alive_actors = sum(1 for a in actors if a is not None)
            start_again = False
            if ray_params.elastic_training:
                if alive_actors < ray_params.num_actors - ray_params.max_failed_actors:
                    raise RuntimeError(
                        "A Ray actor died during training and the maximum "
                        "number of dead actors in elastic training was "
                        "reached. Shutting down training."
                    ) from exc

                # Do not start new actors before resuming training
                # (this might still restart actors during training)
                start_actor_ranks.clear()

                if exc.__cause__ and isinstance(
                    exc.__cause__, RayXGBoostActorAvailable
                ):
                    # New actor available, integrate into training loop
                    logger.info(
                        f"A new actor became available. Re-starting training "
                        f"from latest checkpoint with new actor. "
                        f"This will use {alive_actors} existing actors and "
                        f"start {len(start_actor_ranks)} new actors. "
                        f"Sleeping for 10 seconds for cleanup."
                    )
                    tries -= 1  # This is deliberate so shouldn't count
                    start_again = True

                elif tries + 1 <= max_actor_restarts:
                    if exc.__cause__ and isinstance(
                        exc.__cause__, RayXGBoostTrainingError
                    ):
                        logger.warning(f"Caught exception: {exc.__cause__}")
                    logger.warning(
                        f"A Ray actor died during training. Trying to "
                        f"continue training on the remaining actors. "
                        f"This will use {alive_actors} existing actors and "
                        f"start {len(start_actor_ranks)} new actors. "
                        f"Sleeping for 10 seconds for cleanup."
                    )
                    start_again = True

            elif tries + 1 <= max_actor_restarts:
                if exc.__cause__ and isinstance(exc.__cause__, RayXGBoostTrainingError):
                    logger.warning(f"Caught exception: {exc.__cause__}")
                logger.warning(
                    f"A Ray actor died during training. Trying to restart "
                    f"and continue training from last checkpoint "
                    f"(restart {tries + 1} of {max_actor_restarts}). "
                    f"This will use {alive_actors} existing actors and start "
                    f"{len(start_actor_ranks)} new actors. "
                    f"Sleeping for 10 seconds for cleanup."
                )
                start_again = True

            if start_again:
                time.sleep(5)
                queue.shutdown()
                stop_event.shutdown()
                time.sleep(5)
                queue, stop_event = _create_communication_processes()
            else:
                raise RuntimeError(
                    f"A Ray actor died during training and the maximum number "
                    f"of retries ({max_actor_restarts}) is exhausted."
                ) from exc
            tries += 1

    total_time = time.time() - start_time

    train_additional_results["training_time_s"] = total_training_time
    train_additional_results["total_time_s"] = total_time

    if ray_params.verbose:
        maybe_log = logger.info
    else:
        maybe_log = logger.debug

    maybe_log(
        "[RayXGBoost] Finished XGBoost training on training data "
        "with total N={total_n:,} in {total_time_s:.2f} seconds "
        "({training_time_s:.2f} pure XGBoost training time).".format(
            **train_additional_results
        )
    )

    _shutdown(
        actors=actors,
        pending_actors=pending_actors,
        queue=queue,
        event=stop_event,
        placement_group=pg,
        force=False,
    )

    if isinstance(evals_result, dict):
        evals_result.update(train_evals_result)
    if isinstance(additional_results, dict):
        additional_results.update(train_additional_results)

    return bst





def _predict(model: xgb.Booster, data: RayDMatrix, ray_params: RayParams, **kwargs):
    _assert_ray_support()

    if ray_params.verbose:
        maybe_log = logger.info
    else:
        maybe_log = logger.debug

    if not ray.is_initialized():
        ray.init()

    # Create remote actors
    actors = [
        _create_actor(
            rank=i,
            num_actors=ray_params.num_actors,
            num_cpus_per_actor=ray_params.cpus_per_actor,
            num_gpus_per_actor=ray_params.gpus_per_actor
            if ray_params.gpus_per_actor >= 0
            else 0,
            resources_per_actor=ray_params.resources_per_actor,
            distributed_callbacks=ray_params.distributed_callbacks,
        )
        for i in range(ray_params.num_actors)
    ]
    maybe_log(f"[RayXGBoost] Created {len(actors)} remote actors.")

    # Split data across workers
    wait_load = []
    for actor in actors:
        wait_load.extend(_trigger_data_load(actor, data, []))

    try:
        ray.get(wait_load)
    except Exception as exc:
        logger.warning(f"Caught an error during prediction: {str(exc)}")
        _shutdown(actors, force=True)
        raise

    # Put model into object store
    model_ref = ray.put(model)

    maybe_log("[RayXGBoost] Starting XGBoost prediction.")

    # Train
    fut = [actor.predict.remote(model_ref, data, **kwargs) for actor in actors]

    try:
        actor_results = ray.get(fut)
    except Exception as exc:
        logger.warning(f"Caught an error during prediction: {str(exc)}")
        _shutdown(actors=actors, force=True)
        raise

    _shutdown(actors=actors, force=False)

    return combine_data(data.sharding, actor_results)




@PublicAPI(stability="beta")
def predict(
    model: xgb.Booster,
    data: RayDMatrix,
    ray_params: Union[None, RayParams, Dict] = None,
    _remote: Optional[bool] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    """Distributed XGBoost predict via Ray.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them predict labels
    using an XGBoost booster model. The results are then combined and
    returned.

    Args:
        model: Booster object to call for prediction.
        data: Data object containing the prediction data.
        ray_params: Parameters to configure
            Ray-specific behavior. See :class:`RayParams` for a list of valid
            configuration parameters.
        _remote: Whether to run the driver process in a remote
            function. This is enabled by default in Ray client mode.
        **kwargs: Keyword arguments will be passed to the local
            `xgb.predict()` calls.

    Returns: ``np.ndarray`` containing the predicted labels.

    """
    os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

    if catboost is None:
        raise ImportError(
            "xgboost package is not installed. XGBoost-Ray WILL NOT WORK. "
            'FIX THIS by running `pip install "xgboost-ray"`.'
        )

    if _remote is None:
        _remote = _is_client_connected() and not is_session_enabled()

    if not ray.is_initialized():
        ray.init()

    if _remote:
        return ray.get(
            ray.remote(num_cpus=0)(predict).remote(
                model, data, ray_params, _remote=False, **kwargs
            )
        )

    _maybe_print_legacy_warning()

    ray_params = _validate_ray_params(ray_params)

    max_actor_restarts = (
        ray_params.max_actor_restarts
        if ray_params.max_actor_restarts >= 0
        else float("inf")
    )
    _assert_ray_support()

    if not isinstance(data, RayDMatrix):
        raise ValueError(
            "The `data` argument passed to `train()` is not a RayPool, "
            "but of type {}. "
            "\nFIX THIS by instantiating a RayDMatrix first: "
            "`data = RayDMatrix(data=data)`.".format(type(data))
        )

    tries = 0
    while tries <= max_actor_restarts:
        try:
            return _predict(model, data, ray_params=ray_params, **kwargs)
        except RayActorError:
            if tries + 1 <= max_actor_restarts:
                logger.warning(
                    "A Ray actor died during prediction. Trying to restart "
                    "prediction from scratch. "
                    "Sleeping for 10 seconds for cleanup."
                )
                time.sleep(10)
            else:
                raise RuntimeError(
                    "A Ray actor died during prediction and the maximum "
                    "number of retries ({}) is exhausted.".format(max_actor_restarts)
                )
            tries += 1
    return None



def _autodetect_resources(ray_params, use_method = False):
    cpus_per_actor, gpus_per_actor = _autodetect_resources_base(
        ray_params, use_method
    )
    if ray_params.cpus_per_actor <= 0:
        cpus_per_actor = max(2, cpus_per_actor)
    return cpus_per_actor, gpus_per_actor

    

