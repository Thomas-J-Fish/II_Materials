#!/usr/bin/env python3
# build_caf2.py
# Create CaF2 4x4x4 supercell and write LAMMPS data with type map: 1=Ca, 2=F

from ase.build import bulk
from ase.io import write

# Conventional cubic fluorite cell (Fm-3m)
a = 5.464  # Angstrom
uc = bulk('CaF2', 'fluorite', a=a, cubic=True)   # 12 atoms

# Supercell: 4x4x4 -> 12 * 64 = 768 atoms; ~21.856 Å box length
sc = uc.repeat((4, 4, 4))

# Formal charges
charges = [2.0 if s == 'Ca' else -1.0 for s in sc.get_chemical_symbols()]
sc.set_initial_charges(charges)

# Write LAMMPS data, enforcing species order: 1=Ca, 2=F
write(
    'caf2_4x4x4.data',
    sc,
    format='lammps-data',
    atom_style='charge',
    specorder=['Ca', 'F'],  # => type 1=Ca, type 2=F
)

print("Wrote caf2_4x4x4.data (4x4x4, orthorhombic). Type mapping: 1=Ca, 2=F.")
