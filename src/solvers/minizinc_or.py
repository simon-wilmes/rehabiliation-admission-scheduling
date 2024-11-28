from src.solvers.solver import Solver
from src.instance import Instance
from src.logging import logger
from src.solution import Solution, NO_SOLUTION_FOUND
from typing import List, Dict, Any
import subprocess
import os


class MiniZincSolver(Solver):
    SOLVER_OPTIONS = Solver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update(
        {
            "minizinc_model_file": "treatment_scheduling.mzn",
            "minizinc_solver": "gecode",
            "data_file": "treatment_scheduling.dzn",
        }
    )

    SOLVER_DEFAULT_OPTIONS = {
        "minizinc_model_file": "treatment_scheduling.mzn",
        "minizinc_solver": "gecode",
        "data_file": "treatment_scheduling.dzn",
    }

    def __init__(self, instance: Instance, **kwargs):
        logger.debug(f"Initializing {self.__class__.__name__} with options:")
        for key in self.__class__.SOLVER_DEFAULT_OPTIONS:
            if key in kwargs:
                setattr(self, key, kwargs[key])
                logger.debug(f"  {key} = {kwargs[key]}")
            else:
                setattr(self, key, self.__class__.SOLVER_DEFAULT_OPTIONS[key])
                logger.debug(
                    f"  {key} = {self.__class__.SOLVER_DEFAULT_OPTIONS[key]} (default)"
                )
        super().__init__(instance, **kwargs)

    def _create_model(self):
        # Generate the data file for MiniZinc
        self._generate_data_file()

    def _solve_model(self):
        # Run the MiniZinc model using subprocess
        result = self._run_minizinc_model()
        if result is not None:
            solution = self._parse_solution(result)
            return solution
        else:
            return NO_SOLUTION_FOUND

    def _generate_data_file(self):
        data = self._prepare_data()
        data_str = self._data_to_dzn(data)
        with open(self.data_file, "w") as file:
            file.write(data_str)
        logger.debug(f"Data file '{self.data_file}' generated.")

    def _prepare_data(self) -> Dict[str, Any]:
        # Prepare the data dictionary to be converted into .dzn format
        data = {}

        # Create mappings from IDs to indices starting from 1 (MiniZinc uses 1-based indexing)
        self.p_id_to_idx = {p.id: idx + 1 for idx, p in enumerate(self.P)}
        self.m_id_to_idx = {m.id: idx + 1 for idx, m in enumerate(self.M)}
        self.f_id_to_idx = {f.id: idx + 1 for idx, f in enumerate(self.F)}

        # Sets
        data["P"] = [self.p_id_to_idx[p.id] for p in self.P]
        data["M"] = [self.m_id_to_idx[m.id] for m in self.M]
        data["F"] = [self.f_id_to_idx[f.id] for f in self.F]
        data["D"] = [d for d in self.D]
        data["T"] = [t for t in self.T]

        # Number of time slots per day
        data["num_time_slots_per_day"] = self.num_time_slots_per_day
        data["num_days"] = len(self.D)

        # Treatment durations
        data["du_m"] = [self.du_m[m] for m in self.M]

        # Length of stay for each patient
        data["l_p"] = [self.l_p[p] for p in self.P]

        # Earliest and latest admission days
        data["earliest_admission_day"] = [p.earliest_admission_date.day for p in self.P]
        data["latest_admission_day"] = [p.admitted_before_date.day for p in self.P]

        # Required number of treatments per patient per treatment type
        data["lr_pm"] = self._prepare_lr_pm()

        # Resource availability
        data["av_fdt"] = self._prepare_av_fdt()

        # Bed capacity
        data["beds_capacity"] = self.instance.beds_capacity

        # Max patients per treatment
        data["k_m"] = [self.k_m[m] for m in self.M]

        # Delay and treatment values
        data["delay_value"] = self.delay_value  # type: ignore
        data["treatment_value"] = self.treatment_value  # type: ignore

        # Patients already admitted
        data["already_admitted"] = [p.already_admitted for p in self.P]

        # Number of repetitions per treatment
        data["R_m"] = [self.n_m[m] for m in self.M]

        # Prepare repetition indices
        data["total_reps"], data["m_index"], data["r_index"] = (
            self._prepare_repetition_indices()
        )

        return data

    def _prepare_lr_pm(self) -> List[List[int]]:
        # Prepare the lr_pm array as a 2D array
        lr_pm = []
        for p in self.P:
            p_idx = self.p_id_to_idx[p.id]
            lr_pm_row = []
            for m in self.M:
                m_idx = self.m_id_to_idx[m.id]
                lr = self.lr_pm.get((p, m), 0)
                lr_pm_row.append(lr)
            lr_pm.append(lr_pm_row)
        return lr_pm

    def _prepare_av_fdt(self) -> List[int]:
        av_fdt = []
        for f in self.F:
            f_idx = self.f_id_to_idx[f.id]
            for d in self.D:
                for t in self.T:
                    available = self.av_fdt.get((f, d, t), 0)
                    av_fdt.append(1 if available else 0)
        return av_fdt

    def _prepare_repetition_indices(self):
        m_index = []
        r_index = []
        for m in self.M:
            m_idx = self.m_id_to_idx[m.id]
            num_reps = self.n_m[m]
            for r in range(1, num_reps + 1):
                m_index.append(m_idx)
                r_index.append(r)
        total_reps = len(m_index)
        return total_reps, m_index, r_index

    def _data_to_dzn(self, data: Dict[str, Any]) -> str:
        lines = []
        for key, value in data.items():
            if isinstance(value, list):
                if all(isinstance(v, int) for v in value):
                    line = f"{key} = {value};"
                elif all(isinstance(v, bool) for v in value):
                    bool_str = ["true" if v else "false" for v in value]
                    line = f"{key} = [{', '.join(bool_str)}];"
                elif all(isinstance(v, list) for v in value):  # For 2D arrays
                    flattened = sum(value, [])
                    line = f"{key} = array2d(1..{len(value)}, 1..{len(value[0])}, {flattened});"
                else:
                    line = f"{key} = {value};"
            else:
                line = f"{key} = {value};"
            lines.append(line)
        return "\n".join(lines)

    def _run_minizinc_model(self) -> str | None:
        try:
            cmd = [
                "minizinc",
                "--solver",
                self.minizinc_solver,  # type: ignore
                self.minizinc_model_file,  # type: ignore
                self.data_file,  # type: ignore
            ]
            logger.debug(f"Running MiniZinc model: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug("MiniZinc model solved successfully.")
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error("MiniZinc model failed to solve.")
            logger.error(e.stderr)
            return None

    def _parse_solution(self, result: str) -> Solution:
        # Parse the MiniZinc output and construct the Solution object
        # This method will depend on the output format of your MiniZinc model
        logger.debug("Parsing MiniZinc solution output.")
        import json

        try:
            output_data = json.loads(result)
            # Process output_data to construct the Solution object
            appointments = []
            patients_arrival = {}

            # Extract patient admissions
            for p_idx, admission_day in enumerate(
                output_data["admission_day"], start=1
            ):
                # Map back to patient IDs
                patient_id = [p.id for p in self.P if self.p_id_to_idx[p.id] == p_idx][
                    0
                ]
                patient = next(p for p in self.P if p.id == patient_id)
                patients_arrival[patient] = admission_day

            # Extract treatment schedules
            for i, is_scheduled in enumerate(
                output_data["is_treatment_scheduled"], start=1
            ):
                if is_scheduled:
                    m_idx = output_data["m_index"][i - 1]
                    r = output_data["r_index"][i - 1]
                    start_slot = output_data["start_slot"][i - 1]
                    m_id = [m.id for m in self.M if self.m_id_to_idx[m.id] == m_idx][0]
                    treatment = next(m for m in self.M if m.id == m_id)
                    # Additional data extraction and appointment creation
                    # ...

            solution = Solution(
                instance=self.instance,
                schedule=appointments,
                patients_arrival=patients_arrival,
                solver=self,
                solution_value=output_data.get("objective", None),
            )
            return solution
        except json.JSONDecodeError:
            logger.error("Failed to parse MiniZinc output.")
            return NO_SOLUTION_FOUND
