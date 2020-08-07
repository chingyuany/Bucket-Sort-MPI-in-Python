#!/usr/bin/env python3
# bucketsort.py
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()
N = 64
sorted_array = np.empty(N, dtype="int")

unsorted = np.zeros(N, dtype="int")
proc_count = np.zeros(size, dtype="int")

disp = np.zeros(size, dtype="int")

# generate random numbers for unsorted array
if rank == 0:
    unsorted = np.random.randint(low=0, high=N, size=N)
    print("before sort, unsorted array is below \n", unsorted)

unsorted = comm.bcast(unsorted, 0)

"""
if rank == 1:
    print("rank", rank, "array\n", unsorted)

#  calculate counter
counter = 0

for i in range(0, N):
    if (unsorted[i] >= local_min and unsorted[i] < local_max):
        counter += 1
print("For rank ", rank, "max is", local_max, "min is", local_min,
      "and there are ", counter, " elements in unsorted that falls within this rank\n")

# store to local bucket
local_bucket = np.zeros(counter, dtype="int")
counter = 0
for i in range(0, N):
    if (unsorted[i] >= local_min and unsorted[i] < local_max):
        local_bucket[counter] = unsorted[i]
        counter += 1

"""
local_min = rank * (N/size)
local_max = (rank + 1) * (N/size)
local_bucket = unsorted[np.logical_and(
    unsorted >= local_min, unsorted < local_max)]
counter = np.size(local_bucket)

"""
# Insertion sort for local bucket
for i in range(0, counter):
    for j in range(i+1, counter):
        if (local_bucket[i] > local_bucket[j]):
            tmp = local_bucket[i]
            local_bucket[i] = local_bucket[j]
            local_bucket[j] = tmp
"""
local_bucket.sort()

print("For rank ", rank,  "min is", local_min, "max is", local_max,
      "and there are ", counter, " elements in unsorted that falls within this rank\n")

#print("rank is ", rank, "local array is ", local_bucket, "\n")

# populate proc_count and displacement


proc_count = comm.gather(counter)


if (rank == 0):
    disp[0] = 0
    for i in range(0, size-1):
        disp[i+1] = disp[i] + proc_count[i]

"""
sorted_array_size = np.size(sorted_array)
local_array_size = np.size(local_bucket)

print("rank =", rank, "proc_count =", proc_count,
      "disp =", disp, "sorted_array_size=", sorted_array_size, "local array size =", local_array_size)

      
https://docs.scipy.org/doc/numpy/user/basics.types.html
https://www.mpi-forum.org/docs/mpi-2.2/mpi22-report/node44.htm
mpi.long = numpy.int
"""
comm.Gatherv(local_bucket, [sorted_array, proc_count, disp, MPI.LONG], root=0)

if (rank == 0):
    print("After sort array is\n ", sorted_array, "\n")
