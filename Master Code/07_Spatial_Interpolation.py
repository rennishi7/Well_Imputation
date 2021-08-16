import utils_spatial

# Data Locations
data_root = './Datasets/'
figures_root = './Figures Spatial'
netcdf_filename = 'well_data.nc'
skip_month = 48
grid_steps = 400

# Location must be added
interpolation = utils_spatial.krigging_interpolation(data_root, figures_root)


well_data_dict = interpolation.read_pickle('Well_Data_Imputed', data_root)
well_data = well_data_dict['Data']
coords_df = well_data_dict['Location']
x_c = well_data_dict['Location']['Longitude']
y_c = well_data_dict['Location']['Latitude']

# create grid
grid_x, grid_y = interpolation.create_grid_coords(x_c, y_c, grid_steps)

# extract every nth month of data and transpose array
data_subset = well_data.iloc[::skip_month,:]

# This sets up a netcdf file to store each raster in.
file_nc, raster_data = interpolation.netcdf_setup(grid_x, grid_y, data_subset.index, netcdf_filename)


for i, date in enumerate(data_subset.index):
    values = data_subset.loc[data_subset.index[i]].values

    # fit the model variogram to the experimental variogram
    var_fitted = interpolation.fit_model_var(coords_df, x_c, y_c, values)  # fit variogram
    # when kriging, you need a variogram.
    # the subroutin has a function to plot the variogram and the experimental
    # data so you can review later.

    krig_map = interpolation.krig_field_generate(var_fitted, x_c, y_c, values, grid_x, grid_y) # krig data
    # krig_map.field provides the 2D array of values
    # this function does all the spatial interpolation using the variogram from above.
    # We use the gstool python library availabe on conda (e.g., conda install gstools)

    # write data to netcdf file
    raster_data[i,:,:] = krig_map.field  # add kriged field to netcdf file at time_step

file_nc.close()

print('final')