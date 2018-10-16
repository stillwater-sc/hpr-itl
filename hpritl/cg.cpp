// hpr-itl cg.cpp: HPR Conjugate Gradient algorithms
//
// Copyright (C) 2017-2018 Stillwater Supercomputing, Inc.
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.

#include "common.hpp"

// enable posit arithmetic exceptions
#define POSIT_THROW_ARITHMETIC_EXCEPTION 1
#include <posit>

// MTL
#include <boost/numeric/itl/itl.hpp>
// defines all Krylov solvers
// CG, CGS, BiCG, BiCGStab, BiCGStab2, BiCGStab_ell, FSM, IDRs, GMRES TFQMR, QMR, PC


namespace hpr {
	template<typename Vector, size_t nbits, size_t es, size_t capacity = 10>
	sw::unum::posit<nbits, es> fused_dot(const Vector& x, const Vector& y) {
		sw::unum::quire<nbits, es, capacity> q = 0;
		size_t ix, iy, n = size(x);
		for (ix = 0, iy = 0; ix < n && iy < n; ix = ix + 1, iy = iy + 1) {
			q += sw::unum::quire_mul(x[ix], y[iy]);
		}
		sw::unum::posit<nbits, es> sum;
		convert(q.to_value(), sum);     // one and only rounding step of the fused-dot product
		return sum;
	}

	/// Conjugate Gradients without preconditioning
	template < typename LinearOperator, typename HilbertSpaceX, typename HilbertSpaceB,
		typename Iteration >
	int cg(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b,
			Iteration& iter)
	{
		mtl::vampir_trace<7001> tracer;
		using std::abs; using mtl::conj; using mtl::lazy;
		typedef HilbertSpaceX Vector;
		typedef typename mtl::Collection<HilbertSpaceX>::value_type Scalar;
		typedef typename Iteration::real                            Real;

		constexpr size_t nbits = Scalar::nbits;
		constexpr size_t es = Scalar::es;

		Scalar rho(0), rho_1(0), alpha(0), alpha_1(0);
		Vector p(resource(x)), q(resource(x)), r(resource(x)), z(resource(x));

		r = b - A*x;
		rho = fused_dot<Vector, nbits, es>(r, r);
		while (!iter.finished(Real(sqrt(abs(rho))))) {
			++iter;
			if (iter.first())
				p = r;
			else
				p = r + (rho / rho_1) * p;

			// q = A * p; alpha = rho / dot(p, q);
			(lazy(q) = A * p) || (lazy(alpha_1) = lazy_dot(p, q));
			alpha = rho / alpha_1;

			x += alpha * p;
			rho_1 = rho;
			(lazy(r) -= alpha * q) || (lazy(rho) = lazy_unary_dot(r));
		}

		return iter;
	}
}


int main(int argc, char** argv)
try {
	using namespace std;
	using namespace sw::unum;
	using namespace mtl;
	using namespace mtl::mat;
	using namespace itl;

	bool bSuccess = true;

	const size_t nbits = 32;
	const size_t es = 2;
	const size_t size = 40, N = size * size;

	using Scalar = posit<nbits, es>;
	using Matrix = mtl::mat::compressed2D< Scalar >;
	using Vector = mtl::vec::dense_vector< Scalar >;

	{
		// Create a 1,600 x 1,600 matrix using a 5-point Laplacian stencil
		Matrix A(N, N);
		mtl::mat::laplacian_setup(A, size, size);

		// Create an ILU(0) preconditioner
		pc::ilu_0< Matrix >        P(A);

		// Set b such that x == 1 is solution; start with x == 0
		dense_vector<Scalar>       x(N, 1.0), b(N);
		b = A * x; x = 0;

		// Termination criterion: r < 1e-6 * b or N iterations
		noisy_iteration< Scalar >  iter(b, 500, 1.e-6);

		// Solve Ax == b without a preconditioner P
		itl::cg(A, x, b, iter);
	}


	{
		cout << "HPR Conjugate Gradient\n";

		// Create a 1,600 x 1,600 matrix using a 5-point Laplacian stencil
		Matrix A(N, N);
		mtl::mat::laplacian_setup(A, size, size);

		// Create an ILU(0) preconditioner
		pc::ilu_0< Matrix >        P(A);

		// Set b such that x == 1 is solution; start with x == 0
		dense_vector<Scalar>       x(N, 1.0), b(N);
		b = A * x; x = 0;

		// Termination criterion: r < 1e-6 * b or N iterations
		noisy_iteration< Scalar >  iter(b, 500, 1.e-6);

		// Solve Ax == b without a preconditioner P
		hpr::cg(A, x, b, iter);
	}

	// Solve Ax == b with left preconditioner P
	//itl::cg(A, x, b, iter);

	return (bSuccess ? EXIT_FAILURE : EXIT_SUCCESS);
}
catch (char const* msg) {
	std::cerr << msg << std::endl;
	return EXIT_SUCCESS; //as we manually throwing the not supported yet it should not fall through the cracks     EXIT_FAILURE;
}
catch (const posit_arithmetic_exception& err) {
	std::cerr << "Uncaught posit arithmetic exception: " << err.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const quire_exception& err) {
	std::cerr << "Uncaught quire exception: " << err.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const posit_internal_exception& err) {
	std::cerr << "Uncaught posit internal exception: " << err.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::runtime_error& err) {
	std::cerr << "Uncaught runtime exception: " << err.what() << std::endl;
	return EXIT_FAILURE;
}
catch (...) {
	std::cerr << "Caught unknown exception" << std::endl;
	return EXIT_FAILURE;
}