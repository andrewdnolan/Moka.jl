using CUDA
using Accessors
using NCDatasets
using StructArrays

import Adapt
import KernelAbstractions as KA

# Mesh strucutre comprised of the 
struct Mesh{PCT, DCT, EST}
    PrimaryCells::StructArray{PCT}
    DualCells::StructArray{DCT}
    Edges::StructArray{EST}
end

function Adapt.adapt_structure(backend, x::Mesh)
    return Mesh(Adapt.adapt(backend, x.PrimaryCells), 
                Adapt.adapt(backend, x.DualCells),
                Adapt.adapt(backend, x.Edges))
end

# dimensions of the mesh 
struct dims
    # Number of unique nodes (i.e. cells, edges, vertices)
    Nᶜ::Int 
    Nᵉ::Int
    Nᵛ::Int 
end

###
### Nodes of elements
###

# this is a line segment
struct edge{FT, IT, TWO, ME2} 
    # FT  --> (F)loat (T)ype
    # IT  --> (I)int (T)ype
    # TWO --> (2)
    # ME2 --> (M)ax (E)dges * (2) 

    xᵉ::FT   # X coordinate of edge midpoints in cartesian space
    yᵉ::FT   # Y coordinate of edge midpoints in cartesian space
   
    # intra edge connectivity
    CoE::NTuple{TWO, IT} # (C)ells    (o)n (E)dge
    VoE::NTuple{TWO, IT} # (V)ertices (o)n (E)dge

    # inter edge connectivity
    EoE::NTuple{ME2, IT} # (E)dges    (o)n (E)dge
    WoE::NTuple{ME2, FT} # (W)eights  (o)n (E)dge; reconstruction weights associated w/ EoE

    lₑ::FT
    dₑ::FT 
end

###
### Elements of mesh 
###

# (P)rimary mesh cell
struct Pᵢ{FT, IT, ME} 
    # FT --> (F)loat (T)ype
    # IT --> (I)int (T)ype
    # ME --> (M)ax (E)dges
    
    xᶜ::FT   # X coordinate of cell centers in cartesian space
    yᶜ::FT   # Y coordinate of cell centers in cartesian space
    
    nEoC::IT # (n)umber of (E)dges (o)n (C)ell 

    # intra cell connectivity
    EoC::NTuple{ME, IT} # (E)dges    (o)n (C)ell; set of edges that define the boundary $P_i$
    VoC::NTuple{ME, IT} # (V)ertices (o)n (C)ell

    # inter cell connectivity
    CoC::NTuple{ME, IT} # (C)ells    (o)n (C)ell
    ESoC::NTuple{ME, IT} # (E)dge (S)ign (o)n (C)ell; 

    # area of cell 
    AC::FT
end

# (D)ual mesh cell
struct Dᵢ{FT, IT, VD, ME}
    # FT --> (F)loat (T)ype
    # IT --> (I)int (T)ype
    # VD --> (V)ertex (D)egree
    # ME --> (M)ax (E)dges

    xᵛ::FT 
    yᵛ::FT 
    
    # intra vertex connecivity 
    EoV::NTuple{VD, IT} # (E)dges (o)n (V)ertex; set of edges that define the boundary $D_i$
    CoV::NTuple{VD, IT} # (C)ells (o)n (V)ertex 
    
    # inter vertex connecivity
    ESoV::NTuple{ME, IT} #(E)dge (S)ign (o)n Vertex

    # area of triangle 
    AT::FT
end 

stack(arr, N) = [Tuple(arr[:,i]) for i in 1:N]

function readPrimaryMesh(ds)
    
    # get dimension info 
    nCells  = ds.dim["nCells"] 
    maxEdges = ds.dim["maxEdges"]

    # coordinate data 
    xᶜ = ds["xCell"][:]
    yᶜ = ds["yCell"][:]
    
    nEoC = ds["nEdgesOnCell"][:]
    
    # intra connectivity
    EoC = stack(ds["edgesOnCell"], nCells)
    VoC = stack(ds["verticesOnCell"], nCells)
    # inter connectivity
    CoC = stack(ds["cellsOnCell"], nCells)
    # edge sign on cell, empty for now will be popupated later
    ESoC = stack(zeros(eltype(nEoC), (maxEdges, nCells)), nCells)

    # Cell area
    AC = ds["areaCell"][:]

    StructArray{Pᵢ}((xᶜ = xᶜ, yᶜ = yᶜ, AC = AC,
                     EoC = EoC, VoC = VoC, CoC = CoC,
                     nEoC = nEoC, ESoC = ESoC))
end

function readDualMesh(ds)

    # get dimension info 
    nVertices = ds.dim["nVertices"] 
    maxEdges  = ds.dim["maxEdges"]

    # coordinate data 
    xᵛ = ds["xVertex"][:]
    yᵛ = ds["yVertex"][:]
    
    # intra connectivity
    EoV = stack(ds["edgesOnVertex"], nVertices)
    CoV = stack(ds["cellsOnVertex"], nVertices)
    
    # get the integer type from the NetCDF file
    int_type = eltype(ds["edgesOnVertex"])
    # edge sign on vertex, empty for now will be popupated later
    ESoV = stack(zeros(int_type, (maxEdges, nVertices)), nVertices)

    # Triangle area
    AT = ds["areaTriangle"][:]

    StructArray{Dᵢ}((xᵛ = xᵛ, yᵛ = yᵛ, AT = AT,
                     EoV = EoV, CoV = CoV, ESoV = ESoV))
end 

function readEdgeInfo(ds)

    # coordinate data 
    xᵉ = ds["xEdge"][:]
    yᵉ = ds["yEdge"][:]

    # intra connectivity
    CoE = stack(ds["cellsOnEdge"], ds.dim["nEdges"])
    VoE = stack(ds["verticesOnEdge"], ds.dim["nEdges"])

    # inter connectivity
    EoE = stack(ds["edgesOnEdge"], ds.dim["nEdges"])
    WoE = stack(ds["weightsOnEdge"], ds.dim["nEdges"])

    lₑ = ds["dvEdge"][:]
    dₑ = ds["dcEdge"][:]

        
    StructArray{edge}(xᵉ = xᵉ, yᵉ = yᵉ,
                      CoE = CoE, VoE = VoE, EoE = EoE, WoE = WoE,
                      lₑ = lₑ, dₑ = dₑ)
end

function signIndexField!(primaryMesh::StructArray{Pᵢ}, edges::StructArray{edge})
    
    # create tmp array to store ESoC (b/c struct is immutable)
    edgeSignOnCell = hcat(collect.(primaryMesh.ESoC)...)
    
    # `eachindex`, instead of `enumerate`?
    @inbounds for (iCell, Cell) in enumerate(primaryMesh), i in 1:Cell.nEoC
         
        iEdge = Cell.EoC[i]
        
        # vector points from cell 1 to cell 2
        if iCell == edges[iEdge].CoE[1]
            edgeSignOnCell[i,iCell] = -1
        else 
            edgeSignOnCell[i,iCell] = 1
        end 
    
        # PrimaryCell struct is immutable so need to use Accessor package,
        # convert mutable array to the immutable NTuple type of a Primary Cell 
        @reset Cell.ESoC = typeof(Cell.ESoC)(edgeSignOnCell[:,iCell]) 
        # SoA package creates a view of the parent arrays
        # https://juliaarrays.github.io/StructArrays.jl/stable/counterintuitive
        primaryMesh[iCell] = Cell
    end    
end 

function signIndexField!(dualMesh::StructArray{Dᵢ}, edges::StructArray{edge})
    
    # vertex Degree (3); constant for all dual cells [this is hacky...]
    vertexDegree = length(dualMesh[1].EoV)
    # create tmp array to store ESoC (b/c struct is immutable)
    edgeSignOnVertex = hcat(collect.(dualMesh.ESoV)...)

    @inbounds for (iVertex, Vertex) in enumerate(dualMesh), i in 1:vertexDegree
         
        iEdge = Vertex.EoV[i]
        
        # vector points from cell 1 to cell 2
        if iVertex == edges[iEdge].VoE[1]
            edgeSignOnVertex[i,iVertex] = -1
        else 
            edgeSignOnVertex[i,iVertex] = 1
        end 
    
        # DualCell struct is immutable so need to use Accessor package,
        # convert mutable array to the immutable NTuple type of a Dual Cell 
        @reset Vertex.ESoV = typeof(Vertex.ESoV)(edgeSignOnVertex[:,iVertex]) 
        # SoA package creates a view of the parent arrays
        # https://juliaarrays.github.io/StructArrays.jl/stable/counterintuitive
        dualMesh[iVertex] = Vertex
    end    
end 

function ReadMesh(meshPath::String; backend=KA.CPU())
    
    ds = NCDataset(meshPath, "r", format=:netcdf4)

    PrimaryMesh = readPrimaryMesh(ds)
    DualMesh    = readDualMesh(ds)
    edges       = readEdgeInfo(ds)
    
    # set the edge sign on cells (primary mesh)
    signIndexField!(PrimaryMesh, edges)
    # set the edge sign on vertices (dual mesh)
    signIndexField!(DualMesh, edges)
    
    # adapting SoA to GPUs for mesh classes is as simple as: 
    #PrimaryMesh = Adapt.adapt(CUDABackend(), PrimaryMesh)
    
    mesh = Mesh(PrimaryMesh, DualMesh, edges)

    Adapt.adapt_structure(backend, mesh)
end

