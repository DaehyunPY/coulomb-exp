def norm(arr):
    outp = 0.0
    for a in arr:
        outp += a ** 2.0
    outp **= 0.5
    return outp


class Obj:
    """
    mass(m)
    charge(q)
    location(x,y,z)
    """

    def __init__(self, m, q, r=None, p=None, t=0.0):
        from numpy import array, zeros
        assert m > 0.0
        self.mass = float(m)
        self.charge = float(q)
        if r is None:
            r = array([0.0])
        if p is None:
            p = zeros(len(r))
        assert len(r) == len(p)
        self.dimension = int(len(r))
        self.location = array([float(rr) for rr in r])
        self.momentum = array([float(pp) for pp in p])
        self.time = float(t)

    def __len__(self):
        return len(self.location)

    def move(self, f, dt):
        from numpy import array
        assert len(f) == self.dimension
        f = array(f)
        dp = f * dt
        dr = 0.5 * (2.0 * self.momentum + dp) / self.mass * dt
        self.time += dt
        self.momentum += dp
        self.location += dr
        del dp
        del dr

    def get_interaction_with(self, *objs):
        from numpy import zeros
        n = len(self)
        outp = zeros(n)
        for obj in objs:
            r = self.location - obj.location
            r_abs = norm(r)
            outp += self.charge * obj.charge * r / r_abs ** 3.0
            del r, r_abs
        del n
        return outp

    def get_momentum(self):
        return norm(self.momentum)

    def get_energy(self):
        return self.get_momentum() ** 2.0 / 2.0 / self.mass

    def get_speed(self):
        return self.get_momentum() / self.mass


class SimCoulombExp:
    """

    """

    def __init__(self, *objs):
        self.objects = list()
        self.numbers = 0
        self.dimension = 0
        self.time = 0.0
        self.add_obj(*objs)

    def add_obj(self, *objs):
        if objs is None:
            pass
        else:
            d = objs[0].dimension
            t = objs[0].time
            if len(objs) != 1:
                for obj in objs[1:]:
                    assert d == obj.dimension
                    assert t == obj.time
            if self.numbers == 0:
                self.objects = objs
                self.numbers = len(objs)
                self.dimension = d
                self.time = t
            else:
                self.objects.append(objs)
                self.numbers += len(objs)
                assert d == self.dimension
                assert t == self.time

    def move_objs(self, dt):
        f = []
        for i in range(self.numbers):
            f.append(self.objects[i].get_interaction_with(*(self.objects[:i] + self.objects[i + 1:])))
        for i in range(self.numbers):
            self.objects[i].move(f[i], dt)
        self.time += dt

    def get_total_energies(self):
        poten = 0.0
        for i in range(self.numbers):
            for j in range(i + 1, self.numbers):
                q1 = self.objects[i].charge
                q2 = self.objects[j].charge
                r1 = self.objects[i].location
                r2 = self.objects[j].location
                r = r1 - r2
                r_abd = norm(r)
                poten += q1 * q2 / r_abd
                del q1, q2, r1, r2, r, r_abd
        kinet = 0.0
        for i in range(self.numbers):
            kinet += self.objects[i].get_energy()
        return poten + kinet, poten, kinet


def atomic_to_au(inp):
    from scipy.constants import codata
    return inp * codata.value('atomic mass constant') / codata.value('atomic unit of mass')


def angstrom_to_au(inp):
    from scipy.constants import codata
    return inp * 1.0e-10 / codata.value('atomic unit of length')


def ns_to_au(inp):
    from scipy.constants import codata
    return inp * 1.0e-9 / codata.value('atomic unit of time')


def hartree_to_ev(inp):
    from scipy.constants import codata
    return inp * codata.value('Hartree energy in eV')


def tbc2xy(theta, b, c, m=None):
    if m is None:
        m = [1.0, 1.0, 1.0]
    from numpy import zeros, array, cos, sin, arctan2, dot
    [ma, mb, mc] = m
    xa = zeros(2)
    xb = array([c, 0])
    xc = b * array([cos(theta), sin(theta)])
    com = (ma * xa + mb * xb + mc * xc) / (ma + mb + mc)
    xa -= com
    xb -= com
    xc -= com
    com -= com
    theta = arctan2(*(xc - com))
    rot = array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    xa = dot(rot, xa)
    xb = dot(rot, xb)
    xc = dot(rot, xc)
    del ma, mb, mc, theta, com, rot
    return xa, xb, xc


def abc2xy(a, b, c, m=None):
    from numpy import arccos
    theta = arccos((b ** 2.0 + c ** 2.0 - a ** 2.0) / (2.0 * b * c))
    return tbc2xy(theta, b, c, m)


def get_dt(dt0, p, i):
    return dt0* p ** i

def get_n(t, dt0, p):
    from numpy import log
    return int(log(t / dt0 * (p - 1.0) + 1.0) / log(p))
