//
// Created by 施奕成 on 2023/3/5.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Solver.h"
namespace py = pybind11;

PYBIND11_MODULE(hw1, m) {
  py::class_<Solver>(m, "Solver")
          .def(py::init<>())
          .def("solve", &Solver::solve);
}
