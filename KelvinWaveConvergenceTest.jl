#   Coastal Kelvin Wave Test Case
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

#   simulating a coastal kelvin wave

CODE_ROOT = pwd() * "/"


include(CODE_ROOT * "mode_forward/time_steppers.jl")
include(CODE_ROOT * "mode_init/MPAS_Ocean.jl")
include(CODE_ROOT * "visualization.jl")
include(CODE_ROOT * "mode_init/exactsolutions.jl")

using PyPlot
using PyCall

animation  = pyimport("matplotlib.animation")
ipydisplay = pyimport("IPython.display")

using LinearAlgebra # for norm()
import Dates
using DelimitedFiles


function boundaryCondition2!(mpasOcean, t)
    for iEdge in 1:mpasOcean.nEdges
        if mpasOcean.boundaryEdge[iEdge] == 1
#             mpasOcean.normalVelocityCurrent[iEdge,:] .= 0
            mpasOcean.normalVelocityCurrent[:,iEdge] .=  -kelvinWaveExactNormalVelocity(mpasOcean, iEdge, t)/mpasOcean.nVertLevels# * 5
        end
    end

end


function lateralProfilePeriodic(y)
    return 1e-6*cos.(y/mpasOcean.lX * 4 * pi)
end

function fix_angleedge2!(mpasOcean)
    for iEdge in 1:mpasOcean.nEdges
        if mpasOcean.boundaryEdge[iEdge] == 1
            thisXEdge = mpasOcean.xEdge[iEdge]
            thisYEdge = mpasOcean.yEdge[iEdge]
            cell1 = mpasOcean.cellsOnEdge[1,iEdge]
            cell2 = mpasOcean.cellsOnEdge[2,iEdge]

            xCell1 = mpasOcean.xCell[cell1]
            yCell1 = mpasOcean.yCell[cell1]


                if thisXEdge > xCell1 && abs(thisYEdge - yCell1) < tolerance
                    DeltaX = dcEdge
                    DeltaY = 0.0
                elseif thisXEdge > xCell1 && thisYEdge > yCell1
                    DeltaX = dcEdge/2.0
                    DeltaY = sqrt3over2*dcEdge
                elseif thisXEdge > xCell1 && thisYEdge < yCell1
                    DeltaX = dcEdge/2.0
                    DeltaY = -sqrt3over2*dcEdge
                elseif thisXEdge < xCell1 && abs(thisYEdge - yCell1) < tolerance
                    DeltaX = -dcEdge
                    DeltaY = 0.0
                elseif thisXEdge < xCell1 && thisYEdge > yCell1
                    DeltaX = -dcEdge/2.0
                    DeltaY = sqrt3over2*dcEdge
                elseif thisXEdge < xCell1 && thisYEdge < yCell1
                    DeltaX = -dcEdge/2.0
                    DeltaY = -sqrt3over2*dcEdge
                end
            
            whichedge = findall(mpasOcean.edgesOnCell[:,cell1] .== iEdge)[1]
            
            mpasOcean.angleEdge[iEdge] = returnTanInverseInProperQuadrant(DeltaX,DeltaY) * mpasOcean.edgeSignOnCell[cell1, whichedge]
            if mpasOcean.angleEdge[iEdge] < 0
                mpasOcean.angleEdge[iEdge]  += 2*pi
            end
            
        end
    end
end

function kelvin_test(mesh_directory, base_mesh_file_name, mesh_file_name, periodicity, T, dt, nSaves=1;
        plot=false, animate=false, nvlevels=1)
    mpasOcean = MPAS_Ocean(mesh_directory,base_mesh_file_name,mesh_file_name, periodicity=periodicity, nvlevels=nvlevels)
#     fix_angleedge2!(mpasOcean)
#     fixAngleEdge2!(mpasOcean)
    
    meanCoriolisParameterf = sum(mpasOcean.fEdge) / length(mpasOcean.fEdge)
    meanFluidThicknessH = sum(mpasOcean.bottomDepth)/length(mpasOcean.bottomDepth)
    c = sqrt(mpasOcean.gravity*meanFluidThicknessH)
    rossbyRadiusR = c/meanCoriolisParameterf
    
    println("simulating for T: $T")
    lYedge = maximum(mpasOcean.yEdge) - minimum(mpasOcean.yEdge)

    function lateralProfilePeriodic(y)
        return 1e-6*cos(y/mpasOcean.lY * 4 * pi)
    end
    
    period = lYedge / (4*pi) /c

    lateralProfile = lateralProfilePeriodic
    
    println("generating kelvin wave exact methods for mesh")
    kelvinWaveExactNormalVelocity, kelvinWaveExactSSH, kelvinWaveExactSolution!, boundaryCondition! = kelvinWaveGenerator(mpasOcean, lateralProfile)
    
    println("setting up initial condition")
    kelvinWaveExactSolution!(mpasOcean)
    
    sshOverTimeNumerical = zeros(Float64, (mpasOcean.nCells, nSaves))
    sshOverTimeExact = zeros(Float64, (mpasOcean.nCells, nSaves))
    nvOverTimeNumerical = zeros(Float64, (mpasOcean.nEdges, nSaves))
    nvOverTimeExact = zeros(Float64, (mpasOcean.nEdges, nSaves))
    
    sshOverTimeNumerical[:,1] = dropdims(sum(mpasOcean.layerThickness, dims=1), dims=1) - mpasOcean.bottomDepth
    sshOverTimeExact[:,1] = kelvinWaveExactSSH(mpasOcean, 1:mpasOcean.nCells)
    nvOverTimeNumerical[:,1] = sum(mpasOcean.normalVelocityCurrent, dims=1)
    nvOverTimeExact[:,1] = kelvinWaveExactNormalVelocity(mpasOcean, 1:mpasOcean.nEdges)


    fpath = CODE_ROOT * "output/simulation_convergence/coastal_kelvinwave/$periodicity/CPU/timehorizon_$(T)"
    
    function plotSSHs(frame, desc="", fpath='.')
        fig, axs = plt.subplots(1, 3, figsize=(9,3))
        
        fig, ax = heatMapMesh(mpasOcean, sshOverTimeNumerical[:,frame], fig=fig, ax=axs[1])
        ax.set_title("Numerical Solution")

        fig, ax = heatMapMesh(mpasOcean, sshOverTimeExact[:,frame], fig=fig, ax=axs[2])
        ax.set_title("Exact Solution")
        
        fig, ax = heatMapMesh(mpasOcean, sshOverTimeExact[:,frame] - sshOverTimeNumerical[:,frame], fig=fig, ax=axs[3])#, cMin=-0.005, cMax=0.005)
        ax.set_title("Difference")
        
        fig.suptitle("Coastal Kelvin Wave SSH, $desc")
        
        fig.savefig("$(fpath)/ssh_cell_$(frame).png", bbox_inches="tight")
        
        fig, axs = plt.subplots(1, 3, figsize=(9,3))
        
        fig, ax = edgeHeatMapMesh(mpasOcean, nvOverTimeNumerical[:,frame], fig=fig, ax=axs[1])
        ax.set_title("Numerical Solution")

        fig, ax = edgeHeatMapMesh(mpasOcean, nvOverTimeExact[:,frame], fig=fig, ax=axs[2])
        ax.set_title("Exact Solution")
        
        fig, ax = edgeHeatMapMesh(mpasOcean, nvOverTimeExact[:,frame] - nvOverTimeNumerical[:,frame], fig=fig, ax=axs[3])#, cMin=-0.005, cMax=0.005)
        ax.set_title("Difference")
        
        fig.suptitle("Coastal Kelvin Wave NV, $desc")
        
        fig.savefig("$(fpath)/ssh_edge_$(frame).png", bbox_inches="tight")
        
        return fig
    end
    
    if plot
        
        plotSSHs(1, "Initial Condition", fpath)
        
    end
    
    
    println("original dt $(mpasOcean.dt)")
    nSteps = Int(round(T/mpasOcean.dt/nSaves))
    mpasOcean.dt = T / nSteps / nSaves
    
    
    println("dx $(mpasOcean.dcEdge[1]) \t dt $(mpasOcean.dt) \t dx/c $(maximum(mpasOcean.dcEdge) / c) \t dx/dt $(mpasOcean.dcEdge[1]/mpasOcean.dt)")
    println("period $period \t steps $nSteps")
    
    t = 0
    for i in 1:nSaves
        for j in 1:nSteps
            
             
            calculate_diagnostics!(mpasOcean)
            calculate_normal_velocity_tendency!(mpasOcean)
            update_normal_velocity_by_tendency!(mpasOcean)

            
            t += mpasOcean.dt
            
#             boundaryCondition2!(mpasOcean, t)
            calculate_diagnostics!(mpasOcean)
            calculate_thickness_tendency!(mpasOcean)
            update_thickness_by_tendency!(mpasOcean)
            
        end
        println("t: $t")
        sshOverTimeNumerical[:,i] = dropdims(sum(mpasOcean.layerThickness, dims=1),dims=1) - mpasOcean.bottomDepth
        nvOverTimeNumerical[:,i] = sum(mpasOcean.normalVelocityCurrent, dims=1)
        sshOverTimeExact[:,i] .= kelvinWaveExactSSH(mpasOcean, 1:mpasOcean.nCells, t)
        nvOverTimeExact[:,i] .= kelvinWaveExactNormalVelocity(mpasOcean, 1:mpasOcean.nEdges, t)
    end
    
    
    
    if plot
        
        plotSSHs(nSaves, "T = $(t)", fpath)
        
        if nSaves > 1 && animate
            for frame in 1:nSaves
                plotSSHs(frame, "T = $(frame*nSteps*mpasOcean.dt)")
            end
            
        end
    end
    
    error = sshOverTimeNumerical .- sshOverTimeExact
    MaxErrorNorm = norm(error, Inf)
    L2ErrorNorm = norm(error/sqrt(float(mpasOcean.nCells)))
    
    return mpasOcean.nCells, mpasOcean.dt, MaxErrorNorm, L2ErrorNorm
end

function wrap_regex(str::AbstractString, maxlen = 92)
    replace(str, Regex(".{1,$maxlen}( |\$)") => @s_str "\\0\n")
end

function convergenceplot(nCellsX, errorNorm, normtype, T, decimals, fpath)
    A = [log10.(nCellsX)    ones(length(nCellsX))]
    m, c = A \ log10.(errorNorm)
    y = m*log10.(nCellsX) .+ c
    y = 10 .^ y
    
    slopestr ="$(round(m,digits=decimals))"
    while length(split(slopestr, ".")[end]) < decimals
        slopestr *= "0"
    end

    fig, ax = subplots(1,1, figsize=(9,9))
    tight_layout()
    ax.loglog(nCellsX, errorNorm, label="$normtype Error Norm", marker="s", linestyle="None", color="black")
    ax.loglog(nCellsX, y, label="Best Fit Line, slope=$slopestr", color="black")
    ax.set_title(wrap_regex("Convergence of $normtype Error Norm of Coastal Kelvin Wave, Time Horizon = $(T) s", 50), fontsize=22, fontweight="bold")
    ax.legend(loc="upper right", fontsize=20)
    ax.set_xlabel("Number of cells", fontsize=20)
    ax.set_ylabel("$normtype error norm", fontsize=20)
    ax.grid(which="both")
    fname = "$fpath$(Dates.now())_$(normtype)"
    fig.savefig("$(fname)_convergence.png", bbox_inches="tight")
    
    return fig, ax
end

function convergence_test(periodicity, mesh_directory, operator_name, test, device;
                write_data=false, show_plots=true, decimals=2, resolutions=[64, 128, 256, 512],
                format=(x->string(x)), nvlevels=1)

    nCases = length(resolutions)
    nCellsX = collect(Int.(round.(resolutions)))
    ncells = zeros(Float64, nCases)
    dts = zeros(Float64, nCases)
    MaxErrorNorm = zeros(Float64, nCases)
    L2ErrorNorm = zeros(Float64, nCases)
    
    # calculate maximum dt
    iCase = argmin(resolutions)
    if periodicity == "Periodic"
        base_mesh_file_name = "base_mesh_$(format(nCellsX[iCase])).nc"
    else
        base_mesh_file_name = "culled_mesh_$(format(nCellsX[iCase])).nc"
    end
    mesh_file_name = "mesh_$(format(nCellsX[iCase])).nc"
    mpasOcean = MPAS_Ocean(mesh_directory,base_mesh_file_name,mesh_file_name, periodicity=periodicity, nvlevels=nvlevels)
    
    maxdt = mpasOcean.dt
    T = 32*maxdt
    
    for iCase = 1:nCases
        if periodicity == "Periodic"
            base_mesh_file_name = "base_mesh_$(format(nCellsX[iCase])).nc"
        else
            base_mesh_file_name = "culled_mesh_$(format(nCellsX[iCase])).nc"
        end
        mesh_file_name = "mesh_$(format(nCellsX[iCase])).nc"
        println()
        println("running test $iCase of $nCases, mesh: $mesh_file_name")
        ncells[iCase], dts[iCase], MaxErrorNorm[iCase], L2ErrorNorm[iCase] =
                test(mesh_directory, base_mesh_file_name, mesh_file_name, periodicity, T, maxdt;
                        plot=show_plots, nvlevels=nvlevels)
    end
    
    nCellsX = sqrt.(ncells)
    
    if write_data
        fpath = CODE_ROOT * "output/simulation_convergence/coastal_kelvinwave/$periodicity/$device/timehorizon_$(T)/"
        mkpath(fpath)
        fname = "$fpath$(Dates.now()).txt"
        open(fname, "w") do io
            writedlm(io, [ncells, dts, L2ErrorNorm, MaxErrorNorm])
        end
        println("saved to $fname")
    end
    
    if show_plots
        fpath = CODE_ROOT * "output/simulation_convergence/coastal_kelvinwave/$periodicity/$device/timehorizon_$(T)/"
        convergenceplot(nCellsX, MaxErrorNorm, "Maximum", T, 2, fpath)
        
        convergenceplot(nCellsX, L2ErrorNorm, "\$L^2\$", T, 2, fpath)
    end
end

convergence_test("NonPeriodic_x",
            CODE_ROOT * "/ConvergenceStudyMeshes",
            "Coastal Kelvin Wave",
            kelvin_test, "CPU",
            #resolutions=[32, 64, 144, 216, 324],
            resolutions=[32],
            format=(x->"$(x)x$(x)"),
            write_data=true, show_plots=true, nvlevels=5)
