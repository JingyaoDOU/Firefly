import swiftsimio as sw
import numpy as np
import os
import woma

from .reader import Reader, ParticleGroup


class SWIFTreader(Reader):
    """The custom reader for SWIFT hdf5 snapshot. This reader currently only read gas particles
    data used from planets giant impacts or equilibrations.
    """

    def __init__(
        self,
        snapshotloc,
        UInames=None,
        decimation_factors=None,
        fields=None,
        matid=None,
        npt=None,
        filterFlags=None,
        colormapFlags=None,
        radiusFlags=None,
        logFlags=None,
        **kwargs
    ):
        """Base initialization method for the SWIFTreader instance.
        :snapshotloc: location of the snapshot
        :UInames: UInames: what should the particle groups be called in the webapp UI.
        :decimation_factors: factor by which to reduce the data randomly
            i.e. :code:`data=data[::decimation_factor]`,
            defaults to :code:`[1 for i in ptypes]`
        :matid: material id for different layers. From core to outmost layers.
            i.e. [401, 400, 303, 200] stands for pure-iron, forsterite, ss08 water, HM80 HHe.
        :npt: number of particles in the target planets
        :fields: names of fields to open from snapshot data
            (e.g. Temperature, Pressure, Entropy, internal energy,density, bound).
            The corresponding list would be like : ["T", "P", "S", "U", "RHO","BD"].
        :param filterFlags: flags to signal whether field should be in filter dropdown,
        defaults to [True for i in fields]
        :colormapFlags: flags to signal whether field should be in colormap dropdown,
            defaults to [True for i in fields]
        :radiusFlags: flags to signal whether field should be in radius dropdown,
            defaults to [False for i in fields]
        :logFlags: flags to signal whether the log of the field should be taken,
            defaults to [False for i in fields]

        """
        # set default mateial id read in\
        idoff = 200000000
        mat_dict = {
            400: "mantle",
            400 + idoff: "Impactor_mantle",
            401: "iron",
            401 + idoff: "Impactor_iron",
            402: "alloy",
            402 + idoff: "Impactor_alloy",
            303: "ocean",
            307: "atmosphere",
            200: "atmosphere",
            -1: "all",
        }

        if npt is None:
            npt = 1e12
        if matid is None:
            matid = [401, 400, 303, 200, -1]
        if UInames is None:
            UInames = [mat_dict[i] for i in matid]
        if decimation_factors is None:
            decimation_factors = [1 for i in matid]
        if filterFlags is None:
            filterFlags = [True for i in fields]
        if colormapFlags is None:
            colormapFlags = [True for i in fields]
        if radiusFlags is None:
            radiusFlags = [False for i in fields]
        if logFlags is None:
            logFlags = [False for i in fields]

        lists = [decimation_factors, UInames]
        names = ["decimation_factors", "UInames"]
        for name, llist in zip(names, lists):
            if len(llist) != len(matid):
                raise ValueError(
                    "%s is not the same length as matid (%d,%d)"
                    % (name, len(llist), len(matid))
                )
        ##  fields
        lists = [filterFlags, colormapFlags, radiusFlags, logFlags]
        names = ["filterFlags", "colormapFlags", "radiusFlags", "logFlags"]
        for name, llist in zip(names, lists):
            if len(llist) != len(fields):
                raise ValueError(
                    "%s is not the same length as fields (%d,%d)"
                    % (name, len(llist), len(fields))
                )

        ##  IO/snapshots
        if not os.path.exists(snapshotloc):
            raise FileNotFoundError("Cannot find %s" % snapshotloc)

        self.snapshotloc = snapshotloc
        self.UInames = UInames
        self.decimation_factors = decimation_factors
        self.matid = np.array(matid)
        self.npt = npt
        self.fields = fields
        ## do we want to filter on that attribute?
        self.filterFlags = filterFlags
        ## do we want to color by that attribute?
        self.colormapFlags = colormapFlags
        ## do we want to scale particle size by that attribute?
        self.radiusFlags = radiusFlags
        ## do we need to take the log of it
        self.logFlags = logFlags

        super().__init__(**kwargs)

    def loadData(
        self,
        com=True,
        vcom=True,
    ):
        """Loads SWIFT snapshot data using swiftsimio.
            The coordinates is keep the same as the internal unit you set in the parameter file
            when running simulations.
            Velocity: km/s
            Pressure: Gpa
            Temperature: k
            Entropy: KJ/k/kg
            Internal energy: KJ/kg
            Density: g/cm^3
            smoothing_lengths: same as the internal unit.
            mass of each particle: kg

        :com: if move offset the origin to the center of mass. Default is: True
        :vcom: if offset the velocity. Default is: True
        """
        idoff = 200000000
        eos_dict = {
            400: "ANEOS_forsterite",
            401: "ANEOS_iron",
            402: "ANEOS_Fe85Si15",
            303: "SS08_water",
            307: "CD21_HHe",
            200: "HM80_HHe",
        }

        woma.load_eos_tables(
            [
                eos_dict[i]
                for i in self.matid[np.logical_and(self.matid > 0, self.matid < 1000)]
            ]
        )

        data = sw.load(self.snapshotloc)
        box_mid = 0.5 * data.metadata.boxsize[0]
        pos = np.array(data.gas.coordinates - box_mid)
        # pos = np.array(pos * 1e-3) # convert to km
        data.gas.velocities.convert_to_mks()
        vel = np.array(data.gas.velocities) * 1e-3  # km/s
        h = np.array(data.gas.smoothing_lengths)
        data.gas.masses.convert_to_mks()
        m = np.array(data.gas.masses)
        data.gas.densities.convert_to_mks()
        rho_mks = np.array(data.gas.densities)
        data.gas.densities.convert_to_cgs()
        RHO = np.array(data.gas.densities)
        data.gas.pressures.convert_to_mks()
        P = np.array(data.gas.pressures) * 1e-9
        data.gas.internal_energies.convert_to_mks()
        u_mks = np.array(data.gas.internal_energies)
        U = np.array(data.gas.internal_energies) * 1e-3  # KJ/KG
        matid = np.array(data.gas.material_ids)
        pid = np.array(data.gas.particle_ids)
        T = np.zeros_like(matid)
        S = np.zeros_like(matid)
        T[matid != 200] = woma.eos.eos.A1_T_u_rho(
            u_mks[matid != 200], rho_mks[matid != 200], matid[matid != 200]
        )
        S[matid != 200] = (
            woma.eos.eos.A1_s_u_rho(
                u_mks[matid != 200], rho_mks[matid != 200], matid[matid != 200]
            )
            * 1e-3
        )  # KJ/K/KG
        if any(self.matid > 1000):
            try:
                npt = int(data.gas.npt)
                print("using npt from snapshot")
            except:
                npt = self.npt
                print("using provided npt")
            matid[npt <= pid] += idoff

        if "BD" in self.fields:
            try:
                BD = np.array(data.gas.bound_ids, dtype=int)
                print("loading bound_ids from snapshot")
            except:
                raise TypeError(
                    "This hdf5 file do not have bound_ids and npt written, please write in bound_ids\
                    and npt using function bound_mass"
                )

        if com:
            pos_centerM = np.sum(pos * m[:, np.newaxis], axis=0) / np.sum(m)
            pos -= pos_centerM
        if vcom:
            vel_centerM = np.sum(vel * m[:, np.newaxis], axis=0) / np.sum(m)
            vel -= vel_centerM

        for mid, UIname, dec_factor in list(
            zip(self.matid, self.UInames, self.decimation_factors)
        )[::-1]:
            print("Loading matid %d" % mid)
            sel_matid = matid == mid
            if mid == -1:
                sel_matid = matid > 0

            field_names = []
            field_arrays = []
            field_filter_flags = []
            field_colormap_flags = []
            field_radius_flags = []

            for field, filterFlag, colormapFlag, radiusFlag, logFlag in list(
                zip(
                    self.fields,
                    self.filterFlags,
                    self.colormapFlags,
                    self.radiusFlags,
                    self.logFlags,
                )
            ):

                arr = vars()[field]

                arr = arr[sel_matid]

                if logFlag:
                    arr = np.log10(arr)
                    field = "log10%s" % field

                field_names = np.append(field_names, [field], axis=0)

                field_filter_flags = np.append(field_filter_flags, [filterFlag], axis=0)

                field_colormap_flags = np.append(
                    field_colormap_flags, [colormapFlag], axis=0
                )

                field_radius_flags = np.append(field_radius_flags, [radiusFlag], axis=0)

                field_arrays.append(arr)

            self.particleGroups = np.append(
                self.particleGroups,
                [
                    ParticleGroup(
                        UIname,
                        pos[sel_matid],
                        vel[sel_matid],
                        field_names=field_names,
                        field_arrays=np.array(field_arrays).reshape(
                            -1, pos[sel_matid].shape[0]
                        ),
                        decimation_factor=dec_factor,
                        field_filter_flags=field_filter_flags,
                        field_colormap_flags=field_colormap_flags,
                        field_radius_flags=field_radius_flags,
                    )
                ],
                axis=0,
            )
        for particleGroup in self.particleGroups:
            particleGroup.filenames_opened = data.filename

            ## add this particle group to the reader's settings file
            self.settings.attachSettings(particleGroup)

        self.settings["showParts"]["all"] = 0

        return self.particleGroups
