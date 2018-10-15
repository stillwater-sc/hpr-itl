// hpr-itl bicgstab.cpp: HPR Bi-Conjugate Gradient Stabilized algorithm
//
// Copyright (C) 2017-2018 Stillwater Supercomputing, Inc.
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.

#include "common.hpp"

// enable posit arithmetic exceptions
#define POSIT_THROW_ARITHMETIC_EXCEPTION 1
#include <posit>

// MTL
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
// defines all Krylov solvers
// CG, CGS, BiCG, BiCGStab, BiCGStab2, BiCGStab_ell, FSM, IDRs, GMRES TFQMR, QMR, PC

using namespace mtl;
using namespace itl;

void PositBiCGStab() {
	const size_t nbits = 64;
	const size_t es = 3;
	const size_t size = 40, N = size * size;

	using Scalar = sw::unum::posit<nbits, es>;
	using Matrix = mtl::mat::compressed2D< Scalar >;
	using Vector = mtl::vec::dense_vector< Scalar >;

	// Create a 1,600 x 1,600 matrix using a 5-point Laplacian stencil
	Matrix A(N, N);
	mtl::mat::laplacian_setup(A, size, size);

	// Create an ILU(0) preconditioner
	pc::ilu_0< Matrix >        P(A);

	// Set b such that x == 1 is solution; start with x == 0
	dense_vector<Scalar>          x(N, 1.0), b(N);
	b = A * x; x = 0;

	// Termination criterion: r < 1e-6 * b or N iterations
	noisy_iteration< Scalar >       iter(b, 500, 1.e-6);

	// Solve Ax == b with left preconditioner P
	bicgstab(A, x, b, P, iter);
}

void IEEEBiCGStab() {
	const int size = 40, N = size * size;
	typedef compressed2D<double>  matrix_type;

	// Set up a matrix 1,600 x 1,600 with 5-point-stencil
	matrix_type                   A(N, N);
	mat::laplacian_setup(A, size, size);

	// Create an ILU(0) preconditioner
	pc::ilu_0<matrix_type>        P(A);

	// Set b such that x == 1 is solution; start with x == 0
	dense_vector<double>          x(N, 1.0), b(N);
	b = A * x; x = 0;

	// Termination criterion: r < 1e-6 * b or N iterations
	noisy_iteration<double>       iter(b, 500, 1.e-6);

	// Solve Ax == b with left preconditioner P
	bicgstab(A, x, b, P, iter);
}

int main(int, char**)
{
	IEEEBiCGStab();

	//PositBiCGStab();

    return 0;
}