## -*- coding: utf8 -*-
## Copyright (c) 2014 Lisandro Dalcin, Júlio Hoffimann Mendes
##
## This file is part of HUM.
##
## HUM is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## HUM is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with HUM.  If not, see <http://www.gnu.org/licenses/>.
##
## Created: 10 May 2014
## Author: Júlio Hoffimann Mendes

from mpi4py import MPI

class MPIPool(object):
    """
    MPI-based parallel processing pool

    Design pattern in which a master process distributes tasks to other
    processes (a.k.a. workers) within a MPI communicator.

    Parameters
    ----------
    comm: mpi4py communicator, optional
        MPI communicator for transmitting messages
        Default: MPI.COMM_WORLD

    master: int, optional
        Master process is one of 0, 1,..., comm.size-1
        Default: 0

    debug: bool, optional
        Whether to print debugging information or not
        Default: False

    References
    ----------
    PACHECO, P. S., 1996. Parallel Programming with MPI.
    """
    def __init__(self, comm=MPI.COMM_WORLD, master=0, debug=False):
        assert comm.size > 1, "MPI pool must have at least 2 processes"
        assert 0 <= master < comm.size, "Master process must be in range [0,comm.size)"
        self.comm = comm
        self.master = master
        self.debug = debug
        self.workers = set(range(comm.size))
        self.workers.discard(self.master)


    def is_master(self):
        """
        Returns true if on the master process, false otherwise.
        """
        return self.comm.rank == self.master


    def wait(self):
        """
        Make the workers listen to the master.
        """
        if not self.is_worker(): return
        worker = self.comm.rank
        status = MPI.Status()
        while True:
            if self.debug: print("Worker {0} waiting for task".format(worker))
            task = self.comm.recv(source=self.master, tag=MPI.ANY_TAG, status=status)

            if task is None:
                if self.debug: print("Worker {0} told to quit work".format(worker))
                break

            func, arg = task
            if self.debug: print("Worker {0} got task {1} with tag {2}"
                                 .format(worker, arg, status.tag))

            result = func(arg)

            if self.debug: print("Worker {0} sending answer {1} with tag {2}"
                                 .format(worker, result, status.tag))
            self.comm.ssend(result, self.master, status.tag)


    def map(self, func, iterable):
        """
        Evaluate a function at various points in parallel. Results are
        returned in the requested order (i.e. y[i] = f(x[i])).
        """
        assert self.is_master()

        workerset = self.workers.copy()
        tasklist = [(tid, (func, arg)) for tid, arg in enumerate(iterable)]
        resultlist = [None] * len(tasklist)
        pending = len(tasklist)

        while pending:
            if workerset and tasklist:
                worker = workerset.pop()
                taskid, task = tasklist.pop()
                if self.debug: print("Sent task {0} to worker {1} with tag {2}"
                                     .format(task[1], worker, taskid))
                self.comm.send(task, dest=worker, tag=taskid)

            if tasklist:
                flag = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                if not flag: continue
            else:
                self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

            status = MPI.Status()
            result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.source
            taskid = status.tag
            if self.debug: print("Master received from worker {0} with tag {1}"
                                 .format(worker, taskid))

            workerset.add(worker)
            resultlist[taskid] = result
            pending -= 1

        return resultlist


    def proceed(self):
        """
        Tell all the workers to quit work.
        """
        if not self.is_master(): return
        for worker in self.workers:
            self.comm.send(None, worker, 0)
