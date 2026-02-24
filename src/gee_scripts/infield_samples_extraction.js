// ============================================================================
// In-field sample extraction: per-image S2 + daily ERA5 + elevation
// Exports per-S2-image spectral data, daily meteo, and elevation stats
// for 186 in-field protein sample polygons (20m x 20m).
//
// PREREQUISITE: Upload infield_samples_polygons_20m.geojson to GEE as asset:
//   projects/propane-primacy-481403-u3/assets/infield_samples_polygons_20m
// ============================================================================

var POLYS_ASSET = 'projects/propane-primacy-481403-u3/assets/infield_samples_polygons_20m';
var SCALE = 10;
var TILE_SCALE = 4;
var EXPORT_FOLDER = 'GEE_Infield_Samples';

var MAX_CLOUDY_PIXEL_PCT = 30;
var SCL_BAD = [0, 1, 2, 3, 8, 9, 10, 11];
var SCL_BUFFER = 100;
var EPS = 1e-6;

var S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'SCL'];
var S2_REFL_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];

var INDICES = ['NDVI', 'EVI2', 'GNDVI', 'NDRE', 'CIre', 'IRECI', 'NDWI', 'MSI', 'GCVI',
               'MNDWI', 'NBR', 'NBR2', 'GVI', 'DGCI', 'NDSWIR1RedEdge1', 'TCARI', 'OSAVI',
               'CCCI', 'mARI', 'ExG', 'DBSI', 'SMMI', 'MIRBI', 'CVI', 'S2REP', 'PVI', 'MTCI',
               'SAVI', 'NDRE2', 'RE_ratio', 'PSRI', 'LSWI'];
var RAW_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
var ALL_SPECTRAL = INDICES.concat(RAW_BANDS);

var MONTHS = [
  {year: 2024, month: 7,  name: '2024_07'},
  {year: 2024, month: 8,  name: '2024_08'},
  {year: 2024, month: 9,  name: '2024_09'},
  {year: 2024, month: 10, name: '2024_10'},
  {year: 2024, month: 11, name: '2024_11'},
  {year: 2024, month: 12, name: '2024_12'},
  {year: 2025, month: 1,  name: '2025_01'},
  {year: 2025, month: 2,  name: '2025_02'},
  {year: 2025, month: 3,  name: '2025_03'},
  {year: 2025, month: 4,  name: '2025_04'},
  {year: 2025, month: 5,  name: '2025_05'},
  {year: 2025, month: 6,  name: '2025_06'}
];

var polys = ee.FeatureCollection(POLYS_ASSET);

print('Sample polygons:', polys.size());
print('Months:', MONTHS.length);

// ============================================================================
// Helper functions (identical to wheat_gpc_monthly_simple.js)
// ============================================================================

function safeDiv(num, den) {
  num = ee.Image(num);
  den = ee.Image(den);
  return num.divide(den.where(den.abs().lte(EPS), 1)).updateMask(den.abs().gt(EPS));
}

function toReflectance(img) {
  return img.select(S2_REFL_BANDS).multiply(0.0001).clamp(0, 1.0);
}

function maskClouds(img) {
  var scl = img.select('SCL');
  var bad = ee.Image(0);
  SCL_BAD.forEach(function(v) { bad = bad.or(scl.eq(v)); });
  return img.updateMask(bad.focal_max({radius: SCL_BUFFER, units: 'meters'}).not());
}

function addIndices(img) {
  var B2 = img.select('B2'), B3 = img.select('B3'), B4 = img.select('B4');
  var B5 = img.select('B5'), B6 = img.select('B6'), B7 = img.select('B7');
  var B8 = img.select('B8'), B8A = img.select('B8A');
  var B11 = img.select('B11'), B12 = img.select('B12');

  return img.addBands([
    safeDiv(B8.subtract(B4), B8.add(B4)).clamp(-1,1).rename('NDVI'),
    safeDiv(B8.subtract(B4).multiply(2.5), B8.add(B4.multiply(2.4)).add(1)).clamp(-1,3).rename('EVI2'),
    safeDiv(B8.subtract(B3), B8.add(B3)).clamp(-1,1).rename('GNDVI'),
    safeDiv(B8A.subtract(B5), B8A.add(B5)).clamp(-1,1).rename('NDRE'),
    safeDiv(B8A, B5).subtract(1).clamp(-1,20).rename('CIre'),
    safeDiv(B7.subtract(B4), safeDiv(B5, B6)).clamp(-50,50).rename('IRECI'),
    safeDiv(B8.subtract(B11), B8.add(B11)).clamp(-1,1).rename('NDWI'),
    safeDiv(B11, B8).clamp(0,10).rename('MSI'),
    safeDiv(B8, B3).subtract(1).clamp(-1,20).rename('GCVI'),
    safeDiv(B3.subtract(B11), B3.add(B11)).clamp(-1,1).rename('MNDWI'),
    safeDiv(B8.subtract(B12), B8.add(B12)).clamp(-1,1).rename('NBR'),
    safeDiv(B11.subtract(B12), B11.add(B12)).clamp(-1,1).rename('NBR2'),
    B3.multiply(-0.2848).add(B4.multiply(-0.2435)).add(B8.multiply(0.5436))
      .add(B11.multiply(0.7243)).add(B12.multiply(0.0840)).clamp(-1,1).rename('GVI'),
    safeDiv(B3.subtract(B4), B3.add(B4).add(B2)).multiply(-1).add(1).divide(3).clamp(0,1).rename('DGCI'),
    safeDiv(B5.subtract(B11), B5.add(B11)).clamp(-1,1).rename('NDSWIR1RedEdge1'),
    B5.subtract(B4).multiply(3).subtract(B5.subtract(B3).multiply(0.2).multiply(safeDiv(B5,B4))).clamp(-10,10).rename('TCARI'),
    safeDiv(B8.subtract(B4).multiply(1.16), B8.add(B4).add(0.16)).clamp(-1,2).rename('OSAVI'),
    safeDiv(B8A.subtract(B5), B8A.add(B5)).divide(safeDiv(B8A.subtract(B4), B8A.add(B4)).add(EPS)).clamp(-5,5).rename('CCCI'),
    safeDiv(ee.Image(1), B3).subtract(safeDiv(ee.Image(1), B5)).multiply(B8).clamp(-5,20).rename('mARI'),
    safeDiv(B3.multiply(2).subtract(B4).subtract(B2), B2.add(B3).add(B4).add(EPS)).clamp(-2,2).rename('ExG'),
    safeDiv(B11.subtract(B3), B11.add(B3)).subtract(safeDiv(B8.subtract(B4), B8.add(B4))).clamp(-2,2).rename('DBSI'),
    safeDiv(B8A.subtract(B11), B8A.add(B11)).clamp(-1,1).rename('SMMI'),
    B12.multiply(10).subtract(B11.multiply(9.8)).add(2).clamp(-50,50).rename('MIRBI'),
    safeDiv(B8.multiply(B4), B3.pow(2).add(EPS)).clamp(0,50).rename('CVI'),
    ee.Image(705).add(ee.Image(35).multiply(safeDiv(B4.add(B7).divide(2).subtract(B5), B6.subtract(B5).add(EPS)))).clamp(700,750).rename('S2REP'),
    B8.subtract(B4.multiply(0.355)).subtract(0.149).divide(1.061).clamp(-1,1).rename('PVI'),
    safeDiv(B6.subtract(B5), B5.subtract(B4).add(EPS)).clamp(-30,30).rename('MTCI'),
    safeDiv(B8.subtract(B4), B8.add(B4).add(0.5)).multiply(1.5).clamp(-1,3).rename('SAVI'),
    safeDiv(B7.subtract(B5), B7.add(B5)).clamp(-1,1).rename('NDRE2'),
    safeDiv(B7, B5).clamp(0,10).rename('RE_ratio'),
    safeDiv(B4.subtract(B2), B6.add(EPS)).clamp(-10,10).rename('PSRI'),
    safeDiv(B8.subtract(B11), B8.add(B11)).clamp(-1,1).rename('LSWI')
  ]);
}

// ============================================================================
// PART 1: Per-image Sentinel-2 spectral extraction
// One row per sample per S2 acquisition date
// ============================================================================

print('--- PART 1: Per-image S2 spectral extraction ---');

MONTHS.forEach(function(m) {
  var start = ee.Date.fromYMD(m.year, m.month, 1);
  var end = start.advance(1, 'month');

  // Get all valid S2 images for this month over the study area
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(polys.geometry().bounds())
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUDY_PIXEL_PCT))
    .select(S2_BANDS);

  // For each image, extract values for all sample polygons
  var results = s2.map(function(img) {
    var processed = addIndices(toReflectance(maskClouds(img)));
    var date = img.date().format('YYYY-MM-dd');

    return polys.map(function(f) {
      var geom = f.geometry();
      var stats = processed.select(ALL_SPECTRAL).reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: geom,
        scale: SCALE,
        maxPixels: 1e8,
        bestEffort: true
      });

      return ee.Feature(null, stats)
        .set('sample_key', f.get('sample_key'))
        .set('field_key', f.get('field_key'))
        .set('date', date);
    });
  }).flatten();

  Export.table.toDrive({
    collection: results,
    description: 'infield_daily_s2_' + m.name,
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'infield_daily_s2_' + m.name,
    fileFormat: 'CSV',
    selectors: ['sample_key', 'field_key', 'date'].concat(ALL_SPECTRAL)
  });
});

// ============================================================================
// PART 2: Daily ERA5-Land meteorological extraction
// One row per sample per day
// ============================================================================

print('--- PART 2: Daily ERA5-Land meteo extraction ---');

MONTHS.forEach(function(m) {
  var start = ee.Date.fromYMD(m.year, m.month, 1);
  var end = start.advance(1, 'month');

  var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .filterDate(start, end);

  // For each day, extract meteo for all samples
  var results = era5.map(function(img) {
    var date = img.date().format('YYYY-MM-dd');

    // Compute derived meteo variables from this single day
    var t2m = img.select('temperature_2m').subtract(273.15);
    var t2m_min = img.select('temperature_2m_min').subtract(273.15);
    var t2m_max = img.select('temperature_2m_max').subtract(273.15);
    var precip = img.select('total_precipitation_sum').multiply(1000);
    var pet = img.select('potential_evaporation_sum').multiply(1000).abs();
    var gdd = t2m.max(0);
    var hot_day = t2m_max.gte(30).rename('hot_days');
    var dewpoint = img.select('dewpoint_temperature_2m').subtract(273.15);

    var meteo = ee.Image.cat([
      t2m.rename('t2m_mean'),
      t2m_min.rename('t2m_min'),
      t2m_max.rename('t2m_max'),
      precip.rename('precip'),
      pet.rename('pet'),
      gdd.rename('gdd'),
      hot_day,
      dewpoint.rename('dewpoint')
    ]);

    return polys.map(function(f) {
      var stats = meteo.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: f.geometry(),
        scale: 11132,  // ERA5-Land native resolution (~11km)
        maxPixels: 1e8,
        bestEffort: true
      });

      return ee.Feature(null, stats)
        .set('sample_key', f.get('sample_key'))
        .set('field_key', f.get('field_key'))
        .set('date', date);
    });
  }).flatten();

  var meteoCols = ['sample_key', 'field_key', 'date',
                   't2m_mean', 't2m_min', 't2m_max', 'precip', 'pet',
                   'gdd', 'hot_days', 'dewpoint'];

  Export.table.toDrive({
    collection: results,
    description: 'infield_daily_meteo_' + m.name,
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'infield_daily_meteo_' + m.name,
    fileFormat: 'CSV',
    selectors: meteoCols
  });
});

// ============================================================================
// PART 3: Elevation statistics per sample polygon
// ============================================================================

print('--- PART 3: Elevation stats ---');

var dem = ee.Image('USGS/SRTMGL1_003').select('elevation');
var slope = ee.Terrain.slope(dem);
var aspect = ee.Terrain.aspect(dem);

var elevResults = polys.map(function(f) {
  var geom = f.geometry();

  var elevStats = dem.reduceRegion({
    reducer: ee.Reducer.min().combine(ee.Reducer.max(), null, true)
              .combine(ee.Reducer.mean(), null, true)
              .combine(ee.Reducer.stdDev(), null, true),
    geometry: geom,
    scale: SCALE,
    maxPixels: 1e8
  });

  var slopeStats = slope.reduceRegion({
    reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true),
    geometry: geom,
    scale: SCALE,
    maxPixels: 1e8
  });

  var aspectStats = aspect.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: SCALE,
    maxPixels: 1e8
  });

  return ee.Feature(null, {
    'sample_key': f.get('sample_key'),
    'field_key': f.get('field_key'),
    'elev_min': elevStats.get('elevation_min'),
    'elev_max': elevStats.get('elevation_max'),
    'elev_mean': elevStats.get('elevation_mean'),
    'elev_std': elevStats.get('elevation_stdDev'),
    'slope_mean': slopeStats.get('elevation_mean'),
    'slope_std': slopeStats.get('elevation_stdDev'),
    'aspect_mean': aspectStats.get('elevation')
  });
});

Export.table.toDrive({
  collection: elevResults,
  description: 'infield_elevation_stats',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'infield_elevation_stats',
  fileFormat: 'CSV',
  selectors: ['sample_key', 'field_key', 'elev_min', 'elev_max', 'elev_mean',
              'elev_std', 'slope_mean', 'slope_std', 'aspect_mean']
});

// ============================================================================
// Visualization
// ============================================================================

Map.centerObject(polys, 12);
Map.addLayer(polys, {color: 'red'}, 'In-field sample polygons');
print('Total exports: ' + (MONTHS.length * 2 + 1) + ' (12 S2 + 12 meteo + 1 elevation)');
