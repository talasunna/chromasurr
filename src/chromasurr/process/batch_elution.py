from CADETProcess.processModel import (
    ComponentSystem,
    Langmuir,
    Inlet,
    LumpedRateModelWithoutPores,
    Outlet,
    FlowSheet,
    Process,
)


class BatchElution(Process):
    """
    Preconfigured CADET batch-elution process with Langmuir adsorption,
    adjustable durations, and flowrates for quick prototyping and training.

    Parameters
    ----------
    feed_conc : list of float
        Concentration of components in the feed inlet.
    column_length : float
        Length of the chromatographic column [m].
    axial_dispersion : float
        Axial dispersion coefficient [m^2/s].
    porosity : float
        Total porosity of the packed bed.
    capacity : list of float
        Maximum adsorption capacity for each component [mol/m^3].
    adsorption_rate : list of float
        Adsorption rate constants [1/s].
    desorption_rate : list of float
        Desorption rate constants [1/s].
    feed_duration : float
        Time during which feed is active [s].
    eluent_duration : float
        Time during which eluent is active [s].
    cycle_time : float
        Total cycle time for the simulation [s].
    """

    def __init__(
        self,
        feed_conc=[10.0, 10.0],
        column_length=0.6,
        axial_dispersion=4.7e-7,
        porosity=0.7,
        capacity=[100.0, 100.0],
        adsorption_rate=[0.02, 0.03],
        desorption_rate=[1.0, 1.0],
        feed_duration=60.0,
        eluent_duration=60.0,
        cycle_time=600.0,
    ):
        self.feed_duration_time = feed_duration
        self.eluent_duration_time = eluent_duration

        system = ComponentSystem()
        system.add_component("A")
        system.add_component("B")

        langmuir = Langmuir(system)
        langmuir.is_kinetic = False
        langmuir.capacity = capacity
        langmuir.adsorption_rate = adsorption_rate
        langmuir.desorption_rate = desorption_rate

        feed = Inlet(system, name="feed")
        feed.c = feed_conc
        eluent = Inlet(system, name="eluent")
        eluent.c = [0.0, 0.0]

        column = LumpedRateModelWithoutPores(system, name="column")
        column.binding_model = langmuir
        column.length = column_length
        column.diameter = 0.024
        column.axial_dispersion = axial_dispersion
        column.total_porosity = porosity
        column.solution_recorder.write_solution_bulk = True

        outlet = Outlet(system, name="outlet")

        fs = FlowSheet(system)
        fs.add_unit(feed, feed_inlet=True)
        fs.add_unit(eluent, eluent_inlet=True)
        fs.add_unit(column)
        fs.add_unit(outlet, product_outlet=True)
        fs.add_connection(feed, column)
        fs.add_connection(eluent, column)
        fs.add_connection(column, outlet)

        super().__init__(fs, name="batch_elution")

        Q = 60 / (60 * 1e6)
        self.add_event("feed_on", "flow_sheet.feed.flow_rate", Q)
        self.add_event("feed_off", "flow_sheet.feed.flow_rate", 0.0)
        self.add_event("eluent_on", "flow_sheet.eluent.flow_rate", Q)
        self.add_event("eluent_off", "flow_sheet.eluent.flow_rate", 0.0)

        self.add_duration("feed_duration")
        self.feed_duration.time = feed_duration

        self.add_duration("eluent_duration")
        self.eluent_duration.time = eluent_duration

        self.add_event_dependency("feed_off", ["feed_on", "feed_duration"], [1, 1])
        self.add_event_dependency("eluent_on", ["feed_off"], None)
        self.add_event_dependency("eluent_off",
                                  ["eluent_on", "eluent_duration"], [1, 1]
                                  )
        self.cycle_time = cycle_time
