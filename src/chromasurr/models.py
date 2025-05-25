from CADETProcess.processModel import (
    ComponentSystem,
    Inlet,
    Outlet,
    GeneralRateModel,
    Linear,
    Cstr,
    TubularReactor,
    NoBinding,
    FlowSheet,
    Process
)

def setup_process(process_type: str) -> Process:
    if process_type not in ['a', 'b', 'c', 'd']:
        raise ValueError("Invalid process_type. Must be 'a', 'b', 'c', or 'd'.")

    # Shared components
    component_system = ComponentSystem(1)

    inlet = Inlet(component_system, "inlet")
    inlet.flow_rate = 8.33e-8

    tubing = TubularReactor(component_system, "tubing")
    tubing.length = 0.146  # m
    tubing.diameter = 0.003477898169  # m
    tubing.axial_dispersion = 2.7e-6  # m^2/s

    mixer = Cstr(component_system, "mixer")
    mixer.V = 4.0e-6  # m^3

    outlet = Outlet(component_system, "outlet")

    # Shared column parameters
    column_params = {
        "bed_porosity": 0.35,
        "particle_porosity": 0.33,
        "particle_radius": 4.5e-5,  # m
        "film_diffusion": 2.0e-6,
        "length": 0.25,  # m
        "diameter": 0.01131196598,  # m
        "axial_dispersion": 3.0e-7  # m^2/s
    }

    # Initialize flow sheet
    flow_sheet = FlowSheet(component_system, "flow_sheet")

    def add_common_units_and_connections(flow_sheet, units):
        for unit in units:
            flow_sheet.add_unit(unit)
        for i in range(len(units) - 1):
            flow_sheet.add_connection(units[i], units[i + 1])

    if process_type == 'a':
        # Process a setup
        connector = TubularReactor(component_system, "connector")
        connector.length = 1e-9
        connector.diameter = 1.0
        connector.axial_dispersion = 0.0

        add_common_units_and_connections(flow_sheet, [inlet, tubing, mixer, connector, outlet])

    elif process_type in ['b', 'c', 'd']:
        column = GeneralRateModel(component_system, "column")
        column.bed_porosity = column_params["bed_porosity"]
        column.particle_porosity = column_params["particle_porosity"]
        column.particle_radius = column_params["particle_radius"]
        column.film_diffusion = column_params["film_diffusion"]
        column.length = column_params["length"]
        column.diameter = column_params["diameter"]
        column.axial_dispersion = column_params["axial_dispersion"]

        if process_type == 'b':
            column.pore_diffusion = 0.0  # m^2/s
            column.binding_model = NoBinding()
        elif process_type == 'c':
            column.pore_diffusion = 2.5e-11  # m^2/s
            column.binding_model = NoBinding()
        elif process_type == 'd':
            linear = Linear(component_system)
            linear.adsorption_rate = 4e-4  # 1/s
            linear.desorption_rate = 4e-3  # 1/s
            column.pore_diffusion = 2.5e-11  # m^2/s
            column.binding_model = linear

        column.solution_recorder.write_solution_bulk = True
        column.solution_recorder.write_solution_solid = True

        add_common_units_and_connections(flow_sheet, [inlet, tubing, mixer, column, outlet])


    # Create the process object
    process = Process(flow_sheet, "process")
    process.cycle_time = 500  # Define the cycle time

    # Add events for the simulation
    process.add_event("injectOn", "flow_sheet.inlet.c", 1, 0)  # Start injection at time 0
    process.add_event("injectOff", "flow_sheet.inlet.c", 0, 12)  # Stop injection at time 12

    return process

# %% Setup error models

def add_pump_delay_error(process: Process, delay_range: tuple[float]) -> Process:
    """Add a pump delay error to the process."""


    process_with_error = copy.deepcopy(process)
    pump_delay = rng.uniform(*delay_range)

    process_with_error.events[0].time += pump_delay
    process_with_error.events[1].time += pump_delay
    print(f"Pump delay added: {pump_delay:.2f} seconds")

    return process_with_error

def add_pump_flow_rate_fluctuation(process, fluctuation_range):
    """Add a flow rate fluctuation error to the process."""


    process_with_error = copy.deepcopy(process)
    flow_rate_fluctuation = rng.normal(*fluctuation_range)
    process_with_error.flow_sheet.inlet.flow_rate *= flow_rate_fluctuation
    print(f"Flow rate fluctuation added: {flow_rate_fluctuation:.2f} (relative change)")

    return process_with_error

def add_concentration_variability(process, variability_range):
    """Add concentration variability to the process."""


    process_with_error = copy.deepcopy(process)
    concentration_variability = rng.normal(*variability_range)
    process_with_error.events[0].state *= concentration_variability
    print(f"Concentration variability added: {concentration_variability:.2f} mMol/Mol")

    return process_with_error

def apply_error_models(process, error_models):
    """
    Apply a series of error models to a process.

    Args:
        process: The process object to modify.
        error_models: A list of tuples, each containing an error model function and its arguments.

    Returns:
        Process with all error models applied.
    """
    modified_process = process
    for error_model, args in error_models:
        modified_process = error_model(modified_process, **args)
    return modified_process

def add_noise_to_solution(solution, mean = 1, std_dev = 1e-3):
    """
    Add noise to a given solution.

    Parameters:
        solution (np.ndarray): The solution array to which noise will be added.
        mean (float): Mean of the Gaussian noise.
        std_dev (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: The solution array with added noise.
    """
    noise = np.random.normal(mean, std_dev, len(solution))
    noise = noise.reshape(solution.shape)  # Ensure shape consistency

    noisy_solution = solution * noise  # Apply noise proportionally
    noisy_solution[0] = solution[0]  # Preserve the first value (baseline)

    return noisy_solution

# %% Simulate
process_type = 'b'  # Choose process type ('a', 'b', 'c', or 'd')
process = setup_process(process_type)

from CADETProcess.simulator import Cadet

simulator = Cadet()
#simulator.time_resolution = 0.1


process_without_errors = setup_process(process_type)

simulation_results = simulator.simulate(process_without_errors)

fig, ax = simulation_results.solution.outlet.outlet.plot()

error_models = [ (add_pump_delay_error, {"delay_range": (0, 2)}),
 (add_pump_flow_rate_fluctuation, {"fluctuation_range": (1, 6.5e-3)}),
 (add_concentration_variability, {"variability_range": (1, 9e-3)})
 ]


for i in range(10):
    process_with_error = apply_error_models(process_without_errors, error_models)
    simulation_results = simulator.simulate(process_with_error)
    simulation_results.solution.outlet.outlet.solution = add_noise_to_solution(
        simulation_results.solution.outlet.outlet.solution
    )

    simulation_results.solution.outlet.outlet.plot(ax=ax)

    # Cache the time and concentration values
    #simulation_cache.append({
     #   "time": simulation_results.solution.outlet.outlet.time.copy(),
      #  "concentration": simulation_results.solution.outlet.outlet.solution.copy()
    #})


# Customize the plot
for line in ax.get_lines():  # Access all the lines in the plot
    line.set_linewidth(0.05)  # Set line thickness
    line.set_color('blue')    # Set line color
    line.set_alpha(0.15)

if ax.get_legend() is not None: # Remove the legend
    ax.get_legend().remove()

ax.set_title("Simulation Results (B) - 1000 Iterations ", fontsize=16)
ax.set_xlabel("Time (minutes)", fontsize=14)
ax.set_ylabel("Concentration", fontsize=14)
ax.grid(True)
