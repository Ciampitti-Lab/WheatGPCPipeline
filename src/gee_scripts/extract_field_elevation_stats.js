// Extract elevation, slope, and aspect statistics per field polygon from SRTM

var FIELD_POLYS_ASSET = 'projects/propane-primacy-481403-u3/assets/wheat_polygons_with_geometry';
var SCALE = 10;
var EXPORT_FOLDER = 'GEE_Wheat_Elevation';

var fields = ee.FeatureCollection(FIELD_POLYS_ASSET);
var srtm = ee.Image('USGS/SRTMGL1_003').select('elevation');
var terrain = ee.Terrain.products(srtm);
var slope = terrain.select('slope');
var aspect = terrain.select('aspect');

print('Fields:', fields.size());

var results = fields.map(function(field) {
  var geom = field.geometry();

  var elevStats = srtm.reduceRegion({
    reducer: ee.Reducer.min()
      .combine(ee.Reducer.max(), '', true)
      .combine(ee.Reducer.mean(), '', true)
      .combine(ee.Reducer.stdDev(), '', true)
      .combine(ee.Reducer.median(), '', true)
      .combine(ee.Reducer.percentile([10, 25, 75, 90]), '', true),
    geometry: geom,
    scale: SCALE,
    maxPixels: 1e9,
    bestEffort: true
  });

  var slopeStats = slope.reduceRegion({
    reducer: ee.Reducer.min()
      .combine(ee.Reducer.max(), '', true)
      .combine(ee.Reducer.mean(), '', true)
      .combine(ee.Reducer.stdDev(), '', true),
    geometry: geom,
    scale: SCALE,
    maxPixels: 1e9,
    bestEffort: true
  });

  var aspectStats = aspect.reduceRegion({
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.stdDev(), '', true),
    geometry: geom,
    scale: SCALE,
    maxPixels: 1e9,
    bestEffort: true
  });

  var pixelCount = srtm.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: geom,
    scale: SCALE,
    maxPixels: 1e9,
    bestEffort: true
  });

  var elevMin = ee.Number(elevStats.get('elevation_min'));
  var elevMax = ee.Number(elevStats.get('elevation_max'));

  return field
    .set('elev_min', elevStats.get('elevation_min'))
    .set('elev_max', elevStats.get('elevation_max'))
    .set('elev_mean', elevStats.get('elevation_mean'))
    .set('elev_std', elevStats.get('elevation_stdDev'))
    .set('elev_median', elevStats.get('elevation_median'))
    .set('elev_p10', elevStats.get('elevation_p10'))
    .set('elev_p25', elevStats.get('elevation_p25'))
    .set('elev_p75', elevStats.get('elevation_p75'))
    .set('elev_p90', elevStats.get('elevation_p90'))
    .set('elev_range', elevMax.subtract(elevMin))
    .set('slope_min', slopeStats.get('slope_min'))
    .set('slope_max', slopeStats.get('slope_max'))
    .set('slope_mean', slopeStats.get('slope_mean'))
    .set('slope_std', slopeStats.get('slope_stdDev'))
    .set('aspect_mean', aspectStats.get('aspect_mean'))
    .set('aspect_std', aspectStats.get('aspect_stdDev'))
    .set('pixel_count', pixelCount.get('elevation'));
});

Export.table.toDrive({
  collection: results,
  description: 'field_elevation_stats',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'field_elevation_stats',
  fileFormat: 'CSV'
});

// Per-sample mean elevation/slope/aspect
var samples = ee.FeatureCollection(FIELD_POLYS_ASSET);

var sampleResults = samples.map(function(sample) {
  var geom = sample.geometry();

  var elevMean = srtm.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: geom, scale: SCALE, maxPixels: 1e8
  });
  var slopeMean = slope.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: geom, scale: SCALE, maxPixels: 1e8
  });
  var aspectMean = aspect.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: geom, scale: SCALE, maxPixels: 1e8
  });

  return sample
    .set('elevation', elevMean.get('elevation'))
    .set('slope', slopeMean.get('slope'))
    .set('aspect', aspectMean.get('aspect'));
});

Export.table.toDrive({
  collection: sampleResults,
  description: 'sample_elevation_slope_aspect',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'sample_elevation_slope_aspect',
  fileFormat: 'CSV'
});

Map.centerObject(fields, 9);
Map.addLayer(srtm.clip(fields.geometry().bounds().buffer(5000)),
  {min: 550, max: 800, palette: ['blue', 'green', 'yellow', 'orange', 'red']}, 'SRTM Elevation');
Map.addLayer(ee.Image().paint(fields, 0, 2), {palette: ['white']}, 'Field Outlines');
