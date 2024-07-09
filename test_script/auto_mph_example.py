# %%
import mph
import mph.discovery
import numpy as np
client = mph.start(cores=8,version='6.1')

# %%
# parameters
a = 300e-9
FF = 0.237
r = np.sqrt(FF*a**2/np.pi)
rho_ellipse_0 = 1.6
rotated_angle = 0.0
gamma_phc = 0.2
layer_num_center = 7
layer_num_center_2 = 7


# material
n1 = 3.37
n2 = np.sqrt(3.37**2*(1-gamma_phc)+gamma_phc)

# %%
# create model
try: # prevent overwrite when model already exists
    model = client.create('Model')
    geometries = model/'geometries'
    geometry = geometries.create(2, name='geometry')
except:
    raise Exception("Model already exists")

# CumulativeSelection = model.java.geom(geometry.tag()).selection().create('holes',"CumulativeSelection")
selections = model/'selections'
sel = selections.create('Union', name='holes')
ell_lst = []
for i_x in np.arange(-layer_num_center,layer_num_center+1):
    for i_y in np.arange(-layer_num_center,layer_num_center+1):
        if i_x == 0:
            rotated_angle = np.pi/2
            rotated_angle = np.rad2deg(rotated_angle)
        else:
            rotated_angle = np.arctan(i_y/i_x)
            rotated_angle = np.rad2deg(rotated_angle)
        if i_x**2+i_y**2 <= layer_num_center_2**2:
            rho_ellipse = 1 + (rho_ellipse_0-1)*(i_x**2+i_y**2)/(layer_num_center_2**2)
        else:
            rho_ellipse = rho_ellipse_0
        # rho_ellipse = 1
        rect = geometry.create('Rectangle')
        rect.property('pos', [i_x*a, i_y*a])
        rect.property('size', [a, a])
        rect.property('base', 'center')
        ell = geometry.create('Ellipse')
        ell.property('pos', [i_x*a, i_y*a])
        ell.property('semiaxes', [r/rho_ellipse, r*rho_ellipse])
        ell.property('base', 'center')
        ell.property('rot', rotated_angle)
        ell.property('selresult', 'on')
        ell_lst.append(selections/ell.name())
sel.property('input', ell_lst)
model.build(geometry)
# %%
physics = model/'physics'
ewfd = physics.create('ElectromagneticWavesFrequencyDomain', geometry, name='ewfd')
sct = ewfd.create('Scattering')
sct.property('Order', 'SecondOrder')
sct.select('all')

# %%
materials = model/'materials'
medium1 = materials.create('Common', name='medium 1')
medium1.select('all')
(medium1/'Basic').property('relpermittivity',
    [f'{n1**2}', '0', '0', '0', f'{n1**2}', '0', '0', '0', f'{n1**2}'])

medium2 = materials.create('Common', name='medium 2')
medium2.select(sel)
(medium2/'Basic').property('relpermittivity',
    [f'{n2**2}', '0', '0', '0', f'{n2**2}', '0', '0', '0', f'{n2**2}'])

meshes = model/'meshes'
meshes.create(geometry, name='mesh')
# %%
studies = model/'studies'
solutions = model/'solutions'
study = studies.create(name='Eigenfrequency')
step = study.create('Eigenfrequency', name='Eigenfrequency')
step.property('neigsactive', 'on')
step.property('neigs', 20)
step.property('shift', 'c_const/0.993[um]')
study.java.setGenPlots(False)
study.run()
# %%
plots = model/'plots'
plots.java.setOnlyPlotWhenRequested(True)
plot = plots.create('PlotGroup2D', name='electric field')
plot.property('titletype', 'manual')
plot.property('title', 'Electrostatic field')
plot.property('showlegendsunit', True)
surface = plot.create('Surface', name='field strength')
surface.property('resolution', 'normal')
surface.property('expr', 'ewfd.normE')
arrow = plot.create('ArrowSurface', name='electric field')
arrow.property('expr', ['ewfd.Ex', 'ewfd.Ey'])
arrow.property('xnumber', 5)
arrow.property('ynumber', 5)
plot.property('ispendingzoom', 'on')

# %%
# exports = model/'exports'
# image = exports.create('Image', name='image')
# image.property('sourceobject', plots/'electric field')
# image.property('filename', 'image.png')
# image.property('size', 'manualweb')
# image.property('unit', 'px')
# image.property('height', '720')
# image.property('width', '720')
# image.property('lockratio', 'off')
# image.property('resolution', '96')
# image.property('antialias', 'on')
# image.property('zoomextents', 'on')
# image.property('fontsize', '12')
# image.property('customcolor', [1, 1, 1])
# image.property('background', 'color')
# image.property('gltfincludelines', 'on')
# image.property('title1d', 'on')
# image.property('legend1d', 'on')
# image.property('logo1d', 'on')
# image.property('options1d', 'on')
# image.property('title2d', 'on')
# image.property('legend2d', 'on')
# image.property('logo2d', 'off')
# image.property('options2d', 'on')
# image.property('title3d', 'on')
# image.property('legend3d', 'on')
# image.property('logo3d', 'on')
# image.property('options3d', 'off')
# image.property('axisorientation', 'on')
# image.property('grid', 'on')
# image.property('axes1d', 'on')
# image.property('axes2d', 'on')
# image.property('showgrid', 'on')
# image.property('target', 'file')
# image.property('qualitylevel', '92')
# image.property('qualityactive', 'off')
# image.property('imagetype', 'png')
# image.property('lockview', 'off')

# image.run()
# %%
model.save(r'D:\Desktop\polarization_structure.mph')

# %%