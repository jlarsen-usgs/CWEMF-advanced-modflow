#!/usr/bin/env python
# coding: utf-8

# # Build the watershed model

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import flopy
from flopy.discretization import StructuredGrid

from flopy.utils.gridintersect import GridIntersect
from shapely.geometry import LineString


# In[ ]:


import sys
sys.path.insert(0, "../data")  # add Folder_2 path to search list
from defaults import *


# In[ ]:


base_dir = "./parallel/base"
split_dir = "./parallel/split"
if not os.path.isdir(base_dir):
   os.makedirs(base_dir)
if not os.path.isdir(split_dir):
   os.makedirs(split_dir)


# ### Load the topology

# In[ ]:


fine_topo = flopy.utils.Raster.load("../data/fine_topo.tif")
_ = fine_topo.plot()


# ### Structured grid parameters

# Set the cell dimensions. This will determine the number of cells in the grid. Setting dx = dy = 2500.0 will lead to 9595 active cells

# In[ ]:


dx = dy = 1000.0
nrow = int(Ly / dy) + 1
ncol = int(Lx / dx) + 1


# ### Read in boundary data
# 
# Load the boundary data from `defaults.py` and plot

# In[ ]:


boundary_polygon = string2geom(geometry["boundary"])
bp = np.array(boundary_polygon)

stream_seg = string2geom(geometry["streamseg1"])
stream_seg = stream_seg[::-1]
stream_seg[:2]


# In[ ]:


fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()
ax.set_aspect("equal")

riv_colors = ("blue", "cyan", "green", "orange", "red")

ax.plot(bp[:, 0], bp[:, 1], "ro-")
sa = np.array(stream_seg)
ax.plot(sa[:, 0], sa[:, 1], color=riv_colors[0], lw=0.75, marker="o")


# In[ ]:


stream_seg[:3]


# ### Create a structured grid

# In[ ]:


working_grid = StructuredGrid(
    nlay=1,
    delr=np.full(ncol, dx),
    delc=np.full(nrow, dy),
    xoff=0.0,
    yoff=0.0,
    top=np.full((nrow, ncol), 1000.0),
    botm=np.full((1, nrow, ncol), -100.0),
)

set_structured_idomain(working_grid, boundary_polygon)
print("grid data: ", Lx, Ly, nrow, ncol)


# ### Sample the raw topographic data

# In[ ]:


top_wg = fine_topo.resample_to_grid(
    working_grid,
    band=fine_topo.bands[0],
    method="linear",
    extrapolate_edges=True,
)


# ### Intersect river segments with grid

# In[ ]:


working_grid.top.min()


# In[ ]:


ixs = GridIntersect(
        working_grid,
    )
v = ixs.intersect(LineString(stream_seg), sort_by_cellid=False)
cellids = v['cellids']


# In[ ]:


intersection_rg = np.zeros(working_grid.shape[1:])
for loc in cellids:
    intersection_rg[loc] = 1


# In[ ]:


fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()
pmv = flopy.plot.PlotMapView(modelgrid=working_grid)
ax.set_aspect("equal")
pmv.plot_array(top_wg)
pmv.plot_array(
    intersection_rg,
    masked_values=[
        0,
    ],
    alpha=0.2,
    cmap="Reds_r",
)
pmv.plot_inactive(color_noflow="white")
ax.plot(bp[:, 0], bp[:, 1], "r-")
sa = np.array(stream_seg)
ax.plot(sa[:, 0], sa[:, 1], "b-")


# ### Set the idomain value to 2 where the river intersects the grid

# In[ ]:


river_locations = working_grid.idomain[0].copy()
index = tuple(np.array(list(zip(*cellids))))
river_locations[index] = 2
working_grid.idomain = river_locations.reshape(1, nrow, ncol)

plt.imshow(working_grid.idomain[0])


# ### Define the number of layers and the thickness of layer 1

# In[ ]:


nlay = 5
dv0 = 5.0
# is this reasonable???
leakance = 1.0 / (0.5 * dv0)  # Kv / b  [1/m]


# ### Create the SFR data for the reaches

# In[ ]:


# Build SFR inputs from the stream/grid intersection results in `v`
# Assumes: v, sg, top_wg, leakance already exist
stream_line = LineString(stream_seg)

# Collect and sort intersections along stream direction
_reaches = []
for rec in v:
    i, j = rec["cellids"]
    rlen = float(rec["lengths"])
    midpt = rec["ixshapes"].interpolate(0.5, normalized=True)
    dist = float(stream_line.project(midpt))
    cell_top = float(top_wg[i, j])
    _reaches.append((dist, int(i), int(j), rlen, cell_top))
    
_reaches.sort(key=lambda x: x[0])
_reaches[:2]


# In[ ]:


# regularize stream bed
for r in range(1,len(_reaches)-1):
    bl_prev = _reaches[r-1][4]
    bl = _reaches[r][4]
    bl_next = _reaches[r+1][4]
    if bl > bl_prev:
        # put it in between, or at least flat
        new_bl = min(0.5*(bl_next + bl_prev), bl_prev)
        _reaches[r] = _reaches[r][:4] + (new_bl,)


# In[ ]:


nreaches = len(_reaches)

# Hydraulic parameters (edit if needed)
rwid = 25.0   # reach width
rbth = 1.0    # streambed thickness
rhk = 0.01     # streambed hydraulic conductivity
man = 0.035   # Manning's n

# Build MF6 SFR packagedata and connectiondata
sfr_packagedata = []
sfr_connectiondata = []

for rno, (dist, i, j, rlen, cell_top) in enumerate(_reaches):
    # streambed top
    rtp = cell_top - 1.0

    # slope from next reach (or previous for last reach)
    if rno < nreaches - 1:
        dist2, _, _, _, top2 = _reaches[rno + 1]
        dd = max(dist2 - dist, 1.0)
        dz = max(cell_top - top2, 1e-3)
    elif nreaches > 1:
        dist1, _, _, _, top1 = _reaches[rno - 1]
        dd = max(dist - dist1, 1.0)
        dz = max(top1 - cell_top, 1e-3)
    else:
        dd, dz = 1.0, 1e-3

    rgrd = max(dz / dd, 1e-5)
    ncon = int(rno > 0) + int(rno < nreaches - 1)

    # ifno, cellid, rlen, rwid, rgrd, rtp, rbth, rhk, man, ncon, ustrf, ndv
    sfr_packagedata.append(
        (rno, (0, i, j), rlen, rwid, rgrd, rtp, rbth, rhk, man, ncon, 1.0, 0)
    )

    # Connection convention: upstream positive, downstream negative
    conn = [rno]
    if rno > 0:
        conn.append(rno - 1)
    if rno < nreaches - 1:
        conn.append(-(rno + 1))
    sfr_connectiondata.append(conn)

# Stress period data (set inflow at first reach; adjust as needed)
sfr_perioddata = {0: [(0, "inflow", 0.0)]}

# Initial stage data (set to streambed top for each reach)
sfr_initialstages = [(rno, float(rec[5])) for rno, rec in enumerate(sfr_packagedata)]

# Optional: DRN river proxy using same intersections (used later by ModflowGwfdrn)
# drn_data = []
# for rno, (_, i, j, rlen, cell_top) in enumerate(_reaches):
#     elev = cell_top - 1.0
#     cond = leakance * rlen * rwid
#     drn_data.append(((0, i, j), elev, cond))

print(f"SFR reaches: {nreaches}")
print("Example packagedata row:", sfr_packagedata[0] if nreaches else None)
print("Example connection row:", sfr_connectiondata[0] if nreaches else None)


# In[ ]:


# Plot SFR reach-to-reach connections on topography and flag non-adjacent links

# Build reach -> (row, col) map from sfr_packagedata
reach_rc = {}
for rec in sfr_packagedata:
    rno = int(rec[0])
    _, i, j = rec[1]  # cellid is (k, i, j)
    reach_rc[rno] = (int(i), int(j))

# Build unique connection pairs from sfr_connectiondata
pairs = set()
for conn in sfr_connectiondata:
    rno = int(conn[0])
    for c in conn[1:]:
        cabs = abs(int(c))
        if cabs in reach_rc and rno in reach_rc and cabs != rno:
            pairs.add(tuple(sorted((rno, cabs))))

# Check adjacency and collect bad pairs
bad_pairs = []
for a, b in sorted(pairs):
    ia, ja = reach_rc[a]
    ib, jb = reach_rc[b]
    di, dj = abs(ia - ib), abs(ja - jb)
    is_adjacent = (max(di, dj) == 1)  # adjacent by edge or corner
    if not is_adjacent:
        bad_pairs.append((a, b, (ia, ja), (ib, jb), di, dj))

# Plot
fig, ax = plt.subplots(figsize=figsize)
pmv = flopy.plot.PlotMapView(modelgrid=working_grid, ax=ax)
pmv.plot_array(top_wg)
pmv.plot_inactive(color_noflow="white")
ax.set_aspect("equal")
ax.set_title("SFR connections over topography\n(green=adjacent, red=non-adjacent)")

xcc = working_grid.xcellcenters
ycc = working_grid.ycellcenters

# Draw connection lines
for a, b in sorted(pairs):
    ia, ja = reach_rc[a]
    ib, jb = reach_rc[b]
    xa, ya = xcc[ia, ja], ycc[ia, ja]
    xb, yb = xcc[ib, jb], ycc[ib, jb]
    di, dj = abs(ia - ib), abs(ja - jb)
    color = "lime" if max(di, dj) == 1 else "red"
    ax.plot([xa, xb], [ya, yb], color=color, lw=1.5, alpha=0.9)

# Plot reach centers
rx = [xcc[i, j] for i, j in reach_rc.values()]
ry = [ycc[i, j] for i, j in reach_rc.values()]
ax.scatter(rx, ry, s=8, c="k", zorder=3)

plt.show()

# Report non-adjacent connections
print(f"Total SFR connections checked: {len(pairs)}")
print(f"Non-adjacent connections: {len(bad_pairs)}")
for a, b, rc_a, rc_b, di, dj in bad_pairs[:25]:
    print(f"  {a} ({rc_a}) <-> {b} ({rc_b})  |  drow={di}, dcol={dj}")
if len(bad_pairs) > 25:
    print(f"  ... {len(bad_pairs) - 25} more")


# ### Create the groundwater discharge drain data

# In[ ]:


gw_discharge_data = build_groundwater_discharge_data(
    working_grid,
    leakance,
    top_wg,
)


# ### Create the top and bottom arrays.
# 
# Top array is not used by the model.

# In[ ]:


topc = np.zeros((nlay, nrow, ncol), dtype=float)
botm = np.zeros((nlay, nrow, ncol), dtype=float)
dv = dv0
topc[0] = top_wg.copy()
botm[0] = topc[0] - dv
for idx in range(1, nlay):
    dv *= 1.5
    topc[idx] = botm[idx - 1]
    botm[idx] = topc[idx] - dv


# #### Print the cell thicknesses

# In[ ]:


for k in range(nlay):
    print((topc[k] - botm[k]).mean())


# ### Create idomain and starting head data

# In[ ]:


idomain = np.array([working_grid.idomain[0, :, :].copy() for k in range(nlay)])
strt = np.array([top_wg.copy() for k in range(nlay)], dtype=float)


# ## Build the model files using FloPy
# Note that the CSV solver output is enabled. We will use that in one of the other notebooks.

# In[ ]:


sim = flopy.mf6.MFSimulation(
    sim_ws=base_dir,
    sim_name="sim",
    exe_name="mf6",
    memory_print_option="summary",
)

tdis = flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(100.0,20,1.0)])

gwf = flopy.mf6.ModflowGwf(
    sim,
    modelname="gwf",
    print_input=False,
    save_flows=True,
    newtonoptions="NEWTON UNDER_RELAXATION",
)

imsgwf = flopy.mf6.ModflowIms(
    sim,
    inner_maximum=100,
    outer_maximum=50,
    print_option="ALL",
    outer_dvclose=1e-5,
    inner_dvclose=1e-6,
    linear_acceleration="BICGSTAB",
    filename=f"{gwf.name}.ims",
)
sim.register_ims_package(imsgwf, [gwf.name])

dis = flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=dx,
    delc=dy,
    idomain=idomain,
    top=top_wg,
    botm=botm,
    xorigin=0.0,
    yorigin=0.0,
)

ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)

npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    save_specific_discharge=True,
    icelltype=1,
    k=1.0,
)
sto = flopy.mf6.ModflowGwfsto(
    gwf,
    save_flows=True,
    iconvert=1,
    ss=1e-6,
    sy=0.2,
    steady_state={0: False},
    transient={0: True},
)

rch = flopy.mf6.ModflowGwfrcha(
    gwf,
    recharge=0.0,
)

sfr = flopy.mf6.ModflowGwfsfr(
    gwf,
    filename="sfr.sfr",
    save_flows=True,
    budget_filerecord="sfr.budget.bin",
    print_input=False,    
    stage_filerecord="stages.sfr.bin",
    nreaches=nreaches,
    packagedata=sfr_packagedata,
    connectiondata=sfr_connectiondata,
    initialstages=sfr_initialstages,
    perioddata=sfr_perioddata,
    pname="SFR-1",
)

drn_gwd = flopy.mf6.ModflowGwfdrn(
    gwf,
    auxiliary=["depth"],
    auxdepthname="depth",
    stress_period_data=gw_discharge_data,
    pname="gwd",
    filename="drn_gwd.drn",
)

oc = flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord=f"{gwf.name}.hds",
    budget_filerecord=f"{gwf.name}.cbc",
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    printrecord=[("BUDGET", "ALL")],
)

gwt = flopy.mf6.ModflowGwt(sim, modelname="gwt")

imsgwt = flopy.mf6.ModflowIms(
    sim,
    inner_maximum=100,
    outer_maximum=50,
    print_option="ALL",
    outer_dvclose=1e-5,
    inner_dvclose=1e-6,
    linear_acceleration="BICGSTAB",
    filename=f"{gwt.name}.ims",
)
sim.register_ims_package(imsgwt, [gwt.name])

dis_gwt = flopy.mf6.ModflowGwtdis(
    gwt,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=dx,
    delc=dy,
    idomain=idomain,
    top=top_wg,
    botm=botm,
    xorigin=0.0,
    yorigin=0.0,
)

porosity = 0.2 * np.ones_like(idomain)
mst = flopy.mf6.ModflowGwtmst(
    gwt,
    save_flows=True,
    porosity=porosity,
    pname="mst",
    filename=f"{gwt.name}.mst",
)

ic_gwt = flopy.mf6.ModflowGwtic(gwt, strt=0.0)

adv = flopy.mf6.ModflowGwtadv(gwt, scheme="UTVD")

dsp = flopy.mf6.ModflowGwtdsp(gwt, alh=0.1, alv=0.01, atv=0.01, ath1=0.01, ath2=0.01)

# Instantiate Streamflow Mass Transport package
strm_conc = 10.0
sft_packagedata = []
for irno in range(nreaches):
    t = (irno, strm_conc)
    sft_packagedata.append(t)

sft_perioddata = []
for irno in range(nreaches):
    if irno == 0:
        sft_perioddata.append((irno, "INFLOW", strm_conc))
        
sft = flopy.mf6.ModflowGwtsft(
    gwt,
    pname="sft",
    filename="sft.sft",
    flow_package_name="SFR-1",
    print_input=False,
    packagedata=sft_packagedata,
    reachperioddata=sft_perioddata,
)

ssm = flopy.mf6.ModflowGwtssm(gwt, sources=[[]])

oc_gwt = flopy.mf6.ModflowGwtoc(
    gwt,
    concentration_filerecord=f"{gwt.name}.ucn",
    budget_filerecord=f"{gwt.name}.cbc",
    saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
    printrecord=[("BUDGET", "ALL")],
)

flopy.mf6.ModflowGwfgwt(
    sim,
    exgtype="GWF6-GWT6",
    exgmnamea=gwf.name,
    exgmnameb=gwt.name,
    filename=f"{sim.name}.gwfgwt",
    )


# ### Count the number of active cells

# In[ ]:


ncells, nactive = get_simulation_cell_count(sim)
print("nr. of cells:", ncells, ", active:", nactive)


# ### Write the model files

# In[ ]:


sim.write_simulation()


# ### Run the model

# In[ ]:


#sim.run_simulation()


# ### Plot results

# In[ ]:


# gwf = sim.get_model()
# times = gwf.output.head().get_times()
# base_head = np.squeeze(gwf.output.head().get_data(totim=times[-1]))


# In[ ]:


from flopy.mf6.utils.model_splitter import Mf6Splitter


# In[ ]:


splitter = Mf6Splitter(sim)
mask = splitter.optimize_splitting_mask(nparts=2)
split_sim = splitter.split_multi_model(mask)
split_sim.set_sim_path(split_dir)
split_sim.write_simulation()


# In[ ]:


split_sim.run_simulation()  # processors=2)


# In[ ]:




