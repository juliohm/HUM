// Copyright (c) 2013 Júlio Hoffimann Mendes
//
// This file is part of HUM.
//
// HUM is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// HUM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with HUM.  If not, see <http://www.gnu.org/licenses/>.
//
// Created: 26 Dec 2013
// Author: Júlio Hoffimann Mendes

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>

#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <opm/core/io/vtk/writeVtkData.hpp>
#include <opm/core/linalg/LinearSolverUmfpack.hpp>
#include <opm/core/pressure/IncompTpfa.hpp>
#include <opm/core/pressure/FlowBCManager.hpp>
#include <opm/core/props/IncompPropertiesBasic.hpp>
#include <opm/core/props/IncompPropertiesShadow.hpp>

#include <opm/core/transport/reorder/TransportSolverTwophaseReorder.hpp>

#include <opm/core/simulator/TwophaseState.hpp>
#include <opm/core/simulator/WellState.hpp>

#include <opm/core/utility/miscUtilities.hpp>
#include <opm/core/utility/Units.hpp>
#include <opm/core/wells/WellCollection.hpp>

using namespace Opm;
using namespace unit;
using namespace prefix;

int main (int argc, char* argv[])
{
  if (argc != 2) {
    std::cout << "Usage: simulator realization.dat" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string infile {argv[1]};
  auto pos = infile.rfind(".dat");
  if (pos == std::string::npos) {
    std::cout << "File extension *.dat not found. Stop." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string basename = infile.substr(0,pos);
  std::string outfile {basename + ".out"};

  // grid specs
  int dim = 3, nx = 250, ny = 250, nz = 1;
  double dx = 10*meter, dy = 10*meter, dz = 1*meter;

  GridManager grid_manager(nx, ny, nz, dx, dy, dz);
  const UnstructuredGrid& grid = *grid_manager.c_grid();
  int num_cells = grid.number_of_cells;

  // define the properties of the fluid
  int num_phases = 2; // water, oil

  // density vector (one component per phase)
  std::vector<double> rho {1000*kilogram/cubic(meter), 890*kilogram/cubic(meter)};

  // viscosity vector (one component per phase)
  std::vector<double> mu {0.89*centi*Poise, 1*centi*Poise};

  // porosity and permeability of the rock
  double porosity = 0.3;
  double k = 100*milli*darcy;

  // define the relative permeability function
  // Use a basic fluid description and set this function to be linear.
  // For more realistic fluid, the saturation function is given by the data.
  SaturationPropsBasic::RelPermFunc rel_perm_func = SaturationPropsBasic::Linear;

  // construct a basic fluid with the properties defined above
  IncompPropertiesBasic basic(num_phases, rel_perm_func, rho, mu, porosity, k, dim, num_cells);

  // read permeability map
  std::ifstream realization(infile);
  if (!realization.good()) {
    std::cout << "Input file is not okay. Stop." << std::endl;
    exit(EXIT_FAILURE);
  }

  // skip header
  std::string buff;
  while (realization.peek() == '#') std::getline(realization, buff);

  double facie; std::vector<double> perm_vec(num_cells);
  for (int i = 0; i < num_cells; ++i) {
    realization >> facie;
    perm_vec[i] = 1*facie*darcy + k;
  }

  // overwrite permeability field
  std::vector<double> perm(dim*dim*num_cells);
  for (int i = 0; i < num_cells; ++i)
    for (int j = 0; j < dim-1; ++j)
      perm[dim*dim*i + dim*j + j] = perm_vec[i];

  IncompPropertiesShadow props(basic);
  props.usePermeability(perm.data());

  // gravity field
  const double grav[] {0., 0., -unit::gravity};

  // set up the source term
  // Positive numbers indicate that the cell is a source, while negative numbers indicate a sink.
  std::vector<double> src(num_cells, 0.);

  // compute the pore volume
  std::vector<double> porevol;
  computePorevolume(grid, props.porosity(), porevol);

  // set up the transport solver
  // This is a reordering implicit Euler transport solver.
  double tolerance = 1e-8;
  int max_iterations = 30;
  TransportSolverTwophaseReorder transport_solver(grid, props, NULL, tolerance, max_iterations);

  // time integration parameters
  double dt = 90*day;
  int num_time_steps = 20;

  // define a vector which contains all cell indexes
  // We use this vector to set up parameters on the whole domain.
  std::vector<int> allcells(num_cells);
  std::iota(allcells.begin(), allcells.end(), 0);

  // set up the boundary conditions
  // Letting bcs empty is equivalent to no flow boundary conditions.
  FlowBCManager bcs;

  // initialize water saturation to minimum everywhere
  TwophaseState state;
  state.init(grid, num_phases);
  state.setFirstSat(allcells, props, TwophaseState::MinSat);

  PhaseUsage phase_usage;
  phase_usage.num_phases = num_phases;
  phase_usage.phase_used[BlackoilPhases::Aqua] = 1;
  phase_usage.phase_used[BlackoilPhases::Liquid] = 1;
  phase_usage.phase_used[BlackoilPhases::Vapour] = 0;

  phase_usage.phase_pos[BlackoilPhases::Aqua] = 0;
  phase_usage.phase_pos[BlackoilPhases::Liquid] = 1;

  // This will contain our well-specific information
  WellCollection well_collection;

  // create the production specification for our prod well group
  // We set a target limit for total reservoir rate, and set the controlling
  // mode of the group to be controlled by the reservoir rate.
  ProductionSpecification well_group_prod_spec;
  well_group_prod_spec.reservoir_flow_max_rate_ = 2000*cubic(meter)/day;
  well_group_prod_spec.control_mode_ = ProductionSpecification::RESV;

  // create our prod well group
  // We hand it an empty injection specification, as we don't want to control its injection target.
  std::shared_ptr<WellsGroupInterface>
    prod_well_group(new WellsGroup("prod_group", well_group_prod_spec, InjectionSpecification(), phase_usage));

  // add our well_group to the well_collection
  well_collection.addChild(prod_well_group);

  // create the production specification and Well objects
  // We set all our wells to be group controlled.
  // We pass in the string argument "group" to set the parent group.
  ProductionSpecification production_specification;
  production_specification.control_mode_ = ProductionSpecification::GRUP;

  // auxiliar converter
  auto IDX = [&](int i, int j) { return i-1 + (j-1)*nx; };

  std::vector<std::string> prod_well_names {"prod1","prod2","prod3","prod4","prod5","prod6","prod7","prod8"};
  int prod_well_cells[] {IDX(1,ny), IDX(nx/2,ny), IDX(nx,ny), IDX(1,ny/2), IDX(nx,ny/2), IDX(1,1), IDX(nx/2,1), IDX(nx,1)};

  for (auto wellname : prod_well_names) {
    std::shared_ptr<WellsGroupInterface>
      well_leaf_node(new WellNode(wellname, production_specification, InjectionSpecification(), phase_usage));
    well_collection.addChild(well_leaf_node, "prod_group");
  }

  // create the injection specification for our inj well group
  InjectionSpecification well_group_inj_spec;
  well_group_inj_spec.reservoir_flow_max_rate_ = 2000*cubic(meter)/day;
  well_group_inj_spec.control_mode_ = InjectionSpecification::RESV;

  // create our inj well group.
  std::shared_ptr<WellsGroupInterface> inj_well_group(new WellsGroup("inj_group", ProductionSpecification(), well_group_inj_spec, phase_usage));

  // add to the collection
  well_collection.addChild(inj_well_group);

  // create the injection specification and injection wells
  InjectionSpecification injection_specification;
  injection_specification.control_mode_ = InjectionSpecification::GRUP;

  std::vector<std::string> inj_well_names {"inj1","inj2"};
  int inj_well_cells[] = {IDX(nx/2,ny/4), IDX(nx/2,3*ny/4)};

  for (auto wellname : inj_well_names) {
    std::shared_ptr<WellsGroupInterface> well_leaf_node(new WellNode(wellname, ProductionSpecification(), injection_specification, phase_usage));
    well_collection.addChild(well_leaf_node, "inj_group");
  }

  int num_prod_wells = prod_well_names.size();
  int num_inj_wells  = inj_well_names.size();
  int num_wells = num_prod_wells + num_inj_wells;

  // create the C struct to hold our wells (this is to interface with the solver code)
  Wells* wells = create_wells(num_phases, num_wells, num_wells /*one perforation per well*/);

  // add each well to the C API.
  // To do this we need to specify the relevant cells the well will be located in.
  const double well_index = 1;
  double inj_composition[] = {1., 0.};
  for (int i = 0; i < num_prod_wells; ++i) {
    add_well(PRODUCER, 0., 1, NULL, &prod_well_cells[i], &well_index, prod_well_names[i].c_str(), wells);
  }
  for (int i = 0; i < num_inj_wells; ++i) {
    add_well(INJECTOR, 0., 1, inj_composition, &inj_well_cells[i], &well_index, inj_well_names[i].c_str(), wells);
  }

  // We need to make the well collection aware of our wells object
  well_collection.setWellsPointer(wells);

  // We're not using well controls, just group controls, so we need to apply them.
  well_collection.applyGroupControls();

//  append_well_controls(RESERVOIR_RATE, 1000*cubic(meter)/day, NULL, 8, wells);
//  append_well_controls(RESERVOIR_RATE, 1000*cubic(meter)/day, NULL, 9, wells);

  // set up necessary information for the wells
  WellState well_state;
  well_state.init(wells, state);
  std::vector<double> well_resflowrates_phase;
  std::vector<double> well_surflowrates_phase;
  std::vector<double> fractional_flows;

  // set up the pressure solver.
  LinearSolverUmfpack linsolver;
  IncompTpfa psolver(grid, props, linsolver, grav, wells, src, bcs.c_bcs());

  std::ofstream of(outfile);
  of << "# ";
  for (auto wellname : prod_well_names)
    of << wellname << '\t';
  for (auto wellname : inj_well_names)
    of << wellname << '\t';
  of << std::endl;

  // main loop
  for (int i = 0; i < num_time_steps; ++i) {

    std::ostringstream vtkfilename;
    vtkfilename << basename << "-results-" << std::setw(3) << std::setfill('0') << i << ".vtu";
    std::ofstream vtkfile(vtkfilename.str());
    DataMap dm;
    dm["saturation"]   = &state.saturation();
    dm["pressure"]     = &state.pressure();
    dm["permeability"] = &perm_vec;
    writeVtkData(grid, dm, vtkfile);

    // solving the pressure until the well conditions are met or
    // until reach the maximum number of iterations
    int well_iter = 0, max_well_iterations = 10;
    bool well_conditions_met = false;
    while (!well_conditions_met) {

      // solve the pressure equation
      psolver.solve(dt, state, well_state);

      // compute the new well rates
      // Notice that we approximate (wrongly) surfflowsrates := resflowsrate
      computeFractionalFlow(props, allcells, state.saturation(), fractional_flows);
      computePhaseFlowRatesPerWell(*wells, well_state.perfRates(), fractional_flows, well_resflowrates_phase);
      computePhaseFlowRatesPerWell(*wells, well_state.perfRates(), fractional_flows, well_surflowrates_phase);

      // check if the well conditions are met
      well_conditions_met = well_collection.conditionsMet(well_state.bhp(), well_resflowrates_phase, well_surflowrates_phase);
      ++well_iter;
      if (!well_conditions_met && well_iter == max_well_iterations) {
        OPM_THROW(std::runtime_error, "Conditions not met within " << max_well_iterations << " iterations.");
      }
    }

    // for each well report total rate, production(+) or injection(-)
    const std::vector<double>& well_perfrates = well_state.perfRates();
    for (int w = 0; w < wells->number_of_wells; ++w) {
      double well_rate = 0.;
      for (int perf = wells->well_connpos[w]; perf < wells->well_connpos[w+1]; ++perf) {
        double perf_rate = unit::convert::to(well_perfrates[perf], cubic(meter)/day);
        well_rate += perf_rate;
      }
      of << -well_rate << '\t';
    }
    of << std::endl;

    // transport solver
    double inflow_frac = 1.; // only water injected
    std::vector<double> transport_src;
    computeTransportSource(grid, src, state.faceflux(), inflow_frac, wells, well_state.perfRates(), transport_src);
    transport_solver.solve(&porevol[0], &transport_src[0], dt, state);
  }

  destroy_wells(wells);
}
