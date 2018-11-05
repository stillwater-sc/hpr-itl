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
#define MTL_VERBOSE_ITERATION
#include <boost/numeric/mtl/mtl.hpp>
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

	///  Bi-Conjugate Gradient Stabilized
	template < class LinearOperator, class HilbertSpaceX, class HilbertSpaceB,
		class Preconditioner, class Iteration >
		int bicgstab(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b,
			const Preconditioner& M, Iteration& iter)
	{
		typedef typename mtl::Collection<HilbertSpaceX>::value_type Scalar;
		typedef HilbertSpaceX                                       Vector;
		mtl::vampir_trace<7004> tracer;

		constexpr size_t nbits = Scalar::nbits;
		constexpr size_t es = Scalar::es;

		Scalar     rho_1(0), rho_2(0), alpha(0), beta(0), gamma, omega(0);
		Vector     p(resource(x)), phat(resource(x)), s(resource(x)), shat(resource(x)),
			t(resource(x)), v(resource(x)), r(resource(x)), rtilde(resource(x));

		r = b - A * x;
		rtilde = r;

		while (!iter.finished(r)) {
			++iter;
			rho_1 = fused_dot<Vector, nbits, es>(rtilde, r);
			MTL_THROW_IF(rho_1 == 0.0, itl::unexpected_orthogonality());

			if (iter.first())
				p = r;
			else {
				MTL_THROW_IF(omega == 0.0, itl::unexpected_orthogonality());
				beta = (rho_1 / rho_2) * (alpha / omega);
				p = r + beta * (p - omega * v);
			}
			phat = solve(M, p);
			v = A * phat;

			gamma = fused_dot<Vector, nbits, es>(rtilde, v);
			MTL_THROW_IF(gamma == 0.0, itl::unexpected_orthogonality());

			alpha = rho_1 / gamma;
			s = r - alpha * v;

			if (iter.finished(s)) {
				x += alpha * phat;
				break;
			}
			shat = solve(M, s);
			t = A * shat;
			omega = fused_dot<Vector, nbits, es>(t, s) / fused_dot<Vector, nbits, es>(t, t);

			x += omega * shat + alpha * phat;
			r = s - omega * t;

			rho_2 = rho_1;
		}
		return iter;
	}

} // namespace hpr

template<typename Scalar>
void regular_BiCGStab() {
	using namespace std;
	const size_t size = 40, N = size * size;

	using Matrix = mtl::mat::compressed2D< Scalar >;
	using Vector = mtl::vec::dense_vector< Scalar >;

	// Create a 1,600 x 1,600 matrix using a 5-point Laplacian stencil
	Matrix A(N, N);
	mtl::mat::laplacian_setup(A, size, size);

	// Create an ILU(0) preconditioner
	itl::pc::ilu_0< Matrix >        P(A);

	// Set b such that x == 1 is solution; start with x == 0
	mtl::vec::dense_vector<Scalar>       x(N, 1.0), b(N);
	b = A * x; x = 0;

	// Termination criterion: r < 1e-6 * b or N iterations
	itl::noisy_iteration< Scalar >  iter(b, 5, (Scalar)1.e-6);

	// Solve Ax == b with left preconditioner P
	itl::bicgstab(A, x, b, P, iter);
}

template<typename Scalar>
void fdp_BiCGStab() {
	using namespace std;
	const size_t size = 40, N = size * size;

	using Matrix = mtl::mat::compressed2D< Scalar >;
	using Vector = mtl::vec::dense_vector< Scalar >;

	// Create a 1,600 x 1,600 matrix using a 5-point Laplacian stencil
	Matrix A(N, N);
	mtl::mat::laplacian_setup(A, size, size);

	// Create an ILU(0) preconditioner
	itl::pc::ilu_0< Matrix >        P(A);

	// Set b such that x == 1 is solution; start with x == 0
	mtl::vec::dense_vector<Scalar>       x(N, 1.0), b(N);
	b = A * x; x = 0;

	// Termination criterion: r < 1e-6 * b or N iterations
	itl::noisy_iteration< Scalar >  iter(b, 5, 1.e-6);

	// Solve Ax == b with left preconditioner P
	hpr::bicgstab(A, x, b, P, iter);
}

int main(int, char**)
{
	using namespace std;
	using namespace mtl;

	constexpr size_t nbits = 32;
	constexpr size_t es = 2;

	using Scalar = sw::unum::posit<nbits, es>;

	regular_BiCGStab<float>();
	fdp_BiCGStab< sw::unum::posit<32, 2> >();
	fdp_BiCGStab< sw::unum::posit<24, 3> >();

    return 0;
}

void debugging() {
	using namespace std;
	using namespace mtl;

	constexpr size_t nbits = 32;
	constexpr size_t es = 2;

	using Scalar = sw::unum::posit<nbits, es>;

	{
		size_t nrElements = 4;
		std::vector<double> data = { -0.25, -0.25, 0.25, 0.25 };
		//std::vector<double> data = { 1, -1, 1, -1 };
		vec::dense_vector< double> dv(nrElements);
		vec::dense_vector< Scalar > pv(nrElements);
		{
			double n2 = 0;
			Scalar p2 = 0;
			int i = 0;
			for (auto v : data) {
				dv(i) = v;
				pv(i) = v;
				++i;
				n2 += v*v;
				cout << pretty_print(Scalar(v)*Scalar(v)) << endl;
				p2 += Scalar(v)*Scalar(v);
			}
			cout << "n2 = " << n2 << " sqrt(n2) = " << sqrt(n2) << endl;
			cout << "p2 = " << p2 << " sqrt(p2) = " << sqrt(p2) << endl;
		}
		cout << setprecision(10);
		cout << "data elements      = " << dv << endl;
		cout << "||vector<double>|| = " << two_norm(dv) << endl;
		cout << "||vector<posit>||  = " << two_norm(pv) << endl;

		//Scalar r = two_norm(pv);

		Scalar x = -0.25;
		Scalar a;
		a = abs(x);
		cout << "abs(-0.25) " << a << endl;
	}

	{
		std::vector<double> data = { -0.174422, +0.104777, +0.0490646, +0.0335677, -0.033611, +0.104777, +0.393583, +0.339555, +0.347771, +0.0403458, +0.0490646, +0.339555, +0.305213, +0.354414, +0.0556945, +0.0335677, +0.347771, +0.354414, +0.426084, +0.132617, -0.033611, +0.0403458, +0.0556945, +0.132617, -0.174422 };
		vec::dense_vector< double> dv(25);
		vec::dense_vector< Scalar > pv(25);

		// || r || = +44.4951

		{
			int i = 0;
			for (auto v : data) {
				dv(i) = v;
				pv(i) = v;
				++i;
			}
		}
		cout << "||vector<double>|| = " << two_norm(dv) << endl;
		cout << "||vector<posit>||  = " << two_norm(pv) << endl;

		Scalar r = two_norm(pv);
	}
}