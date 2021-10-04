using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
root = 0
commsize = MPI.Comm_size(comm)

include("mode_init/MPAS_Ocean.jl")
include("mode_init/MPAS_OceanHalos.jl")

println("loading ocean from file...")

cd("/global/homes/r/rstrauss/repos/MPAS_Ocean_Julia/")
fullOcean = MPAS_Ocean("MPAS_O_Shallow_Water/Mesh+Initial_Condition+Registry_Files/Periodic", "base_mesh.nc", "mesh.nc", periodicity="Periodic")

halowidth = 5

cellsInChunk, edgesInChunk, verticesInChunk, cellsFromChunk, cellsToChunk = divide_ocean(fullOcean, halowidth, 2, 2)#; iChunk = rank+1)

myCells = cellsInChunk[rank+1] # cellsedgesvertices[1]
myEdges = edgesInChunk[rank+1] # cellsedgesvertices[2]
myVertices = verticesInChunk[rank+1] # cellsedgesvertices[3]

myOcean = mpas_subset(fullOcean, myCells, myEdges, myVertices)

MPI.Barrier(comm)

if rank == root
	println("ocean distributed between ranks.")
end

############### Ocean is now distributed between nodes. Let's do some simulation ####################

# set up initial condition

include("mode_init/initial_conditions.jl")

gaussianInit!(myOcean)

MPI.Barrier(comm)

if rank == root
	println("initial condition set.")
end

# simulate for a while

include("mode_forward/time_steppers.jl")

nFrames = 30

mpasOcean = myOcean

sshOverTime = zeros(nFrames+1, mpasOcean.nCells)
sshOverTime[1,:] = mpasOcean.sshCurrent

for f in 1:nFrames
	# simulate until the halo areas are all invalid and need to be updated
	for h in 1:halowidth
		forward_backward_step!(mpasOcean)
	end
	
	### request cells in my halo from chunks with those cells
	halobufferssh = [] # temporarily stores new halo ssh
	halobuffernv = [] # temporarily stores new halo normal velocity
	recreqs = []
	for (srcchunk, localcells) in cellsFromChunk[rank+1]	
		newhalossh = Array{eltype(mpasOcean.sshCurrent)}(undef, length(localcells))
		append!(halobufferssh, [newhalossh])
		reqssh = MPI.Irecv!(newhalossh, srcchunk-1, 0, comm) # tag 0 for ssh
		append!(recreqs, [reqssh])

		localedges = collect(Set(mpasOcean.edgesOnCell[:,localcells]))
		newhalonv = Array{eltype(mpasOcean.normalVelocityCurrent)}(undef, length(localedges))
		append!(halobuffernv, [newhalonv])
		reqnv = MPI.Irecv!(newhalonv, srcchunk-1, 1, comm) # tag 1 for norm vel
		append!(recreqs, [reqnv])
	end
	
	MPI.Barrier(comm)
	### send cells in main non-halo area to chunks that need them for their halo
	sendreqs = []
	for (dstchunk, localcells) in cellsToChunk[rank+1]
		reqssh = MPI.Isend(mpasOcean.sshCurrent[localcells], dstchunk-1, 0, comm)
		append!(sendreqs, [reqssh])
		
		localedges = collect(Set(mpasOcean.edgesOnCell[:,localcells])) # Set to remove duplicates
		order = sortperm(myEdges[localedges])
		reqnv = MPI.Isend(mpasOcean.normalVelocityCurrent[localedges[order]], dstchunk-1, 1, comm)
		append!(sendreqs, [reqnv])
	end
	
	### copy the recieved data into the ocean's halo
	if rank == root
	# 	println("halo buffer before: ", halobuffernv[1][1:10])
	end
	MPI.Barrier(comm)
	MPI.Waitall!([recreqs..., sendreqs...])
	if rank == root
	# 	println("halo buffer after: ", halobuffernv[1][1:10])
	end
	MPI.Barrier(comm)
	for (i, (_, localcells)) in enumerate(cellsFromChunk[rank+1])
		mpasOcean.sshCurrent[localcells] = halobufferssh[i]
		localedges = collect(Set(mpasOcean.edgesOnCell[:,localcells]))
		order = sortperm(myEdges[localedges])
		mpasOcean.normalVelocityCurrent[localedges[order]] = halobuffernv[i]
	end

	sshOverTime[f+1,:] = mpasOcean.sshCurrent

	if rank == root
		println("iteration $f of $nFrames complete")
	end
end

if rank == root
	println("simulation complete, writing output arrays")
end

using DelimitedFiles

open("out/dist_test_1_ssh_rank_$rank.txt", "w") do io
	writedlm(io, sshOverTime)
end

println("$rank: file written")
