function ocn_init(Config_filepath)
    
    # read the configuration file 
    Config = ConfigRead(Config_filepath)

    #TO DO: Read the constants ?? 


    # setup the mesh 
    Mesh = ocn_setup_mesh(Config)

    # setup clock 
    Clock = ocn_setup_clock(Config)
    
    # return the model setup instance
    Setup = ModelSetup(Config, Mesh, Clock)

    # Prognostics should be intialized here, 
    # add option to read from input file (i.e. mesh) or from restrart
    Prog = PrognosticVars_init(Config, Mesh)
    
    # Diagnostic and Tendecies probabl don't need to be initialized here 
    # instead should happen within the `ocn_run` method, prior to entering 
    # the first time integration loop to ensure values are initialized 
    Diag = DiagnosticVars_init(Config, Mesh) 

    Tend = TendencyVars(Config, Mesh)

    return Setup, Diag, Tend, Prog
end 


function ocn_setup_mesh(Config::GlobalConfig)
    # get mesh section of the streams file
    meshConfig = ConfigGet(Config.streams, "mesh")
    # get mesh filepath from streams section
    mesh_fp = ConfigGet(meshConfig, "filename_template")
    # read the inut mesh from the configuartion file 
    # NOTE: This might be a restart file based on config options 
    ReadMesh(mesh_fp)
end 

function ocn_setup_clock(Config::GlobalConfig)

    # Get the nested Config objects 
    time_managementConfig = ConfigGet(Config.namelist, "time_management")
    time_integrationConfig = ConfigGet(Config.namelist, "time_integration")
    
    dt = ConfigGet(time_integrationConfig, "config_dt")
    stop_time = ConfigGet(time_managementConfig, "config_stop_time")
    start_time = ConfigGet(time_managementConfig, "config_start_time")
    run_duration = ConfigGet(time_managementConfig, "config_run_duration")
    restart_timestamp_name = ConfigGet(time_managementConfig, "config_restart_timestamp_name")

    ## I think I need some example of this, becuase not immediately obvious to me how to 
    ## deal with this. Or really, what this actually would look like. 
    #if restart_timestamp_name == "file"
    #
    #else 
    #  
    #end 
    
    # both config_run_duration and config_stop_time specified. Ensure that values are consitent


    if run_duration != "none" 
        clock = mpas_create_clock(dt, start_time; runDuration=run_duration)
        if stop_time != "none" 
            start_time + run_duration != stop_time && println("Warning: config_run_duration and config_stop_time are inconsitent: using config_run_duration.")
        else 
            stop_time = start_time + run_duration
        end 
    elseif stop_time != "none"
        clock = mpas_create_clock(dt, start_time; stopTime=stop_time)
    else 
        throw("Error: Neither config_run_duration nor config_stop_time were specified.")
    end
    
    # create the end of simulation alarm 
    simulationAlarm = OneTimeAlarm("simulation_end", stop_time)
        
    # attached the simulation_end alarm to the clock 
    attachAlarm!(clock, simulationAlarm)

    return clock
end 

