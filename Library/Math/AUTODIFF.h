# pragma once
//#####################################################################
// Class AUTODIFF
//#####################################################################
#include <utility>
#include <Math/VECTOR.h>

namespace JGSL {

template <class T, class DiffT>
inline auto make_auto_diff(const T& s, const DiffT& ds);

template <class T, class DiffT>
class AUTODIFF {
public:
    T s;
    DiffT ds;

    AutoDiff()
    {
    }

    AutoDiff(const T& s, const DiffT& ds)
            : s(s)
            , ds(ds)
    {
    }

    template <class OtherDiffT>
    AutoDiff(const AutoDiff<T, OtherDiffT>& x)
            : s(x.s)
            , ds(x.ds)
    {
    }

    AutoDiff& operator=(const T& t)
    {
        s = t;
        ds.setZero();
        return *this;
    }

    template <class OtherDiffT>
    AutoDiff& operator=(const AutoDiff<T, OtherDiffT>& x)
    {
        s = x.s;
        ds = x.ds;
        return *this;
    }

    static constexpr AutoDiff Zero()
    {
        return AutoDiff((T)0);
    }

    template <class OtherDiffT>
    auto operator+(const AutoDiff<T, OtherDiffT>& x) const
    {
        return make_auto_diff(s + x.s, ds + x.ds);
    }

    template <class OtherDiffT>
    AutoDiff& operator+=(const AutoDiff<T, OtherDiffT>& x)
    {
        s += x.s;
        ds += x.ds;
        return *this;
    }

    template <class OtherDiffT>
    auto operator-(const AutoDiff<T, OtherDiffT>& x) const
    {
        return make_auto_diff(s - x.s, ds - x.ds);
    }

    template <class OtherDiffT>
    auto operator*(const AutoDiff<T, OtherDiffT>& x) const
    {
        return make_auto_diff(s * x.s, ds * x.s + x.ds * s);
    }

    template <class OtherDiffT>
    auto operator/(const AutoDiff<T, OtherDiffT>& x) const
    {
        T one_over_x = (T)1 / x.s;
        return make_auto_diff(s * one_over_x, (ds * x.s - x.ds * s) * one_over_x * one_over_x);
    }

    auto operator+(const T& x) const
    {
        return make_auto_diff(s + x, ds);
    }

    AutoDiff& operator+=(const T& x)
    {
        s += x.s;
        return *this;
    }

    auto operator-(const T& x) const
    {
        return make_auto_diff(s - x, ds);
    }

    auto operator*(const T& x) const
    {
        return make_auto_diff(s * x, ds * x);
    }

    auto operator/(const T& x) const
    {
        T one_over_x = (T)1 / x;
        return make_auto_diff(s * one_over_x, ds * one_over_x);
    }

    auto operator-() const
    {
        return make_auto_diff(-s, -ds);
    }

    template <class OtherDiffT>
    bool operator<(const AutoDiff<T, OtherDiffT>& x) const
    {
        return s < x.s;
    }

    template <class OtherDiffT>
    bool operator<=(const AutoDiff<T, OtherDiffT>& x) const
    {
        return s <= x.s;
    }

    template <class OtherDiffT>
    bool operator==(const AutoDiff<T, OtherDiffT>& x) const
    {
        return s == x.s;
    }

    template <class OtherDiffT>
    bool operator>(const AutoDiff<T, OtherDiffT>& x) const
    {
        return s > x.s;
    }

    template <class OtherDiffT>
    bool operator>=(const AutoDiff<T, OtherDiffT>& x) const
    {
        return s >= x.s;
    }

    bool operator<(const T& x) const
    {
        return s < x;
    }

    bool operator<=(const T& x) const
    {
        return s <= x;
    }

    bool operator>(const T& x) const
    {
        return s > x;
    }

    bool operator>=(const T& x) const
    {
        return s >= x;
    }
};

template <class T, int num_vars = 1>
using ADScalar = AutoDiff<T, VECTOR<T, num_vars>>;

template <class T, int dim, int num_vars = dim>
using ADVec = VECTOR<ADScalar<T, num_vars>, dim>;

//TODO: implement ADMat

template <class T, class DiffT>
inline auto make_auto_diff(const T& s, const DiffT& ds)
{
    return AutoDiff<T, DiffT>(s, ds);
}

template <int num_vars, class T>
inline auto var(const T& s, int var_id)
{
    return make_auto_diff(s, VECTOR<T, num_vars>::Unit_Vector(var_id));
}

template <class T, int dim>
inline auto vars(const VECTOR<T, dim>& x)
{
    VECTOR<AutoDiff<T, Vector<T, dim>>, dim> v;
    for (int i = 0; i < x.size(); i++)
        v(i) = var<dim>(x(i), i);
    return v;
}

//TODO: implement second vars

//TODO: implement fillJacobian

template <class T, class DiffT>
auto operator+(const T& s, const AutoDiff<T, DiffT>& x)
{
    return make_auto_diff(s + x.s, x.ds);
}

template <class T, class DiffT>
auto operator-(const T& s, const AutoDiff<T, DiffT>& x)
{
    return make_auto_diff(s - x.s, -x.ds);
}

template <class T, class DiffT>
auto operator*(const T& s, const AutoDiff<T, DiffT>& x)
{
    return make_auto_diff(s * x.s, s * x.ds);
}

template <class T, class DiffT>
auto operator/(const T& s, const AutoDiff<T, DiffT>& x)
{
    T one_over_x = (T)1 / x.s;
    T s_over_x = s * one_over_x;
    return make_auto_diff(s_over_x, -s_over_x * one_over_x * x.ds);
}

template <class T, class OtherDiffT>
bool operator<(const T& s, const AutoDiff<T, OtherDiffT>& x)
{
    return s < x.s;
}

template <class T, class OtherDiffT>
bool operator<=(const T& s, const AutoDiff<T, OtherDiffT>& x)
{
    return s <= x.s;
}

template <class T, class OtherDiffT>
bool operator>(const T& s, const AutoDiff<T, OtherDiffT>& x)
{
    return s > x.s;
}

template <class T, class OtherDiffT>
bool operator>=(const T& s, const AutoDiff<T, OtherDiffT>& x)
{
    return s >= x.s;
}

template <class T, class DiffT>
inline auto conj(const AutoDiff<T, DiffT>& x)
{
    return x;
}
template <class T, class DiffT>
inline auto real(const AutoDiff<T, DiffT>& x)
{
    return x;
}

template <class T, class DiffT>
inline auto imag(const AutoDiff<T, DiffT>&)
{
    return AutoDiff<T, DiffT>::Zero();
}
template <class T, class DiffT>
inline auto abs(const AutoDiff<T, DiffT>& x)
{
    T sign = x.s < 0 ? -1 : 1;
    return sign * x;
}
template <class T, class DiffT>
inline auto abs2(const AutoDiff<T, DiffT>& x)
{
    return x * x;
}

template <class T, class DiffT>
inline auto sqrt(const AutoDiff<T, DiffT>& x)
{
    using std::sqrt;
    T sqrt_x = sqrt(x.s);
    T scale = sqrt_x == (T)0 ? (T)0 : 1 / (2 * sqrt_x); // Wrong but no nan
    return make_auto_diff(sqrt_x, scale * x.ds);
}

template <class T, class DiffT1, class DiffT2>
inline auto pow(const AutoDiff<T, DiffT1>& x, const AutoDiff<T, DiffT2>& y)
{
    using std::log;
    using std::pow;
    T z = pow(x.s, y.s);
    return make_auto_diff(z, z * ((y.s * x.ds) / x.s + y.ds * log(x.s)));
}

template <class T, class DiffT>
inline auto pow(const T& x, const AutoDiff<T, DiffT>& y)
{
    using std::log;
    using std::pow;
    T z = pow(x, y.s);
    return make_auto_diff(z, z * log(x) * y.ds);
}

template <class T, class DiffT>
inline auto pow(const AutoDiff<T, DiffT>& x, const T& y)
{
    using std::log;
    using std::pow;
    T z = pow(x.s, y - 1);
    return make_auto_diff(z * x.s, y * z * x.ds);
}

template <class T, class DiffT>
inline auto sin(const AutoDiff<T, DiffT>& x)
{
    using std::cos;
    using std::sin;
    return make_auto_diff(sin(x.s), cos(x.s) * x.ds);
}

template <class T, class DiffT>
inline auto cos(const AutoDiff<T, DiffT>& x)
{
    using std::cos;
    using std::sin;
    return make_auto_diff(cos(x.s), -sin(x.s) * x.ds);
}

template <class T, class DiffT>
inline auto tan(const AutoDiff<T, DiffT>& x)
{
    using std::cos;
    using std::tan;
    T sec = (T)1 / cos(x.s);
    return make_auto_diff(tan(x.s), x.ds * sec * sec);
}

template <class T, class DiffT1, class DiffT2>
inline auto atan2(const AutoDiff<T, DiffT1>& x, const AutoDiff<T, DiffT2>& y)
{
    using std::atan2;
    return make_auto_diff(atan2(x.s, y.s), (y.s * x.ds - x.s * y.ds) / (x.s * x.s + y.s * y.s));
}

template <class T, class DiffT>
inline auto exp(const AutoDiff<T, DiffT>& x)
{
    using std::exp;
    T exps = exp(x.s);
    return make_auto_diff(exps, exps * x.ds);
}

template <class T, class DiffT>
inline auto log(const AutoDiff<T, DiffT>& x)
{
    using std::log;
    return make_auto_diff(log(x.s), x.ds / x.s);
}

template <class T, class DiffT>
inline std::ostream& operator<<(std::ostream& os, const AutoDiff<T, DiffT>& x)
{
    Eigen::IOFormat fmt(4, 0, " ", "\n", "(", ")");
    return os << x.s << ' ' << x.ds.transpose().format(fmt);
}

}