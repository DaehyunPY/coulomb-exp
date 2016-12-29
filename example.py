import numpy as np
import CoulombExp as exp

massNe = 20.1797
massKr = 83.798
(posA, posB, posC) = exp.tbc2xy(67.01, 3.68, 3.68, [massNe, massKr, massKr])
posA = np.append(posA, 0.0)
posB = np.append(posB, 0.0)
posC = np.append(posC, 0.0)

t0 = 0.0
t1 = 1e10
dt0 = 1e-3
p = 1.0+1e-4
n = exp.get_n(t1-t0, dt0, p)

aNe = exp.Obj(exp.atomic_to_au(massNe), 1.0, exp.angstrom_to_au(posA), np.zeros(3), t0)
bKr = exp.Obj(exp.atomic_to_au(massKr), 1.0, exp.angstrom_to_au(posB), np.zeros(3), t0)
cKr = exp.Obj(exp.atomic_to_au(massKr), 1.0, exp.angstrom_to_au(posC), np.zeros(3), t0)
sim = exp.SimCoulombExp(aNe, bKr, cKr)

tE0, pE0, kE0 = sim.get_total_energies()
for i in range(n):
    sim.move_objs(exp.get_dt(dt0, p, i))
    tE, pE, kE = sim.get_total_energies()
    print(sim.time, pE/tE*100, tE/tE0)

print([exp.hartree_to_ev(o.get_energy()) for o in sim.objects])
print([exp.hartree_to_ev(e) for e in [tE0, pE0, kE0]])
print([exp.hartree_to_ev(e) for e in sim.get_total_energies()])
