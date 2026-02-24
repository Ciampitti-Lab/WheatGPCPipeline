// Monthly S2 + ERA5 extraction for wheat GPC prediction
// Exports one CSV per month (Jul 2024 - Jun 2025), aggregate to periods in Python

var POLYS_ASSET = 'projects/propane-primacy-481403-u3/assets/wheat_polygons_with_geometry';
var SCALE = 10;
var TILE_SCALE = 8;
var EXPORT_FOLDER = 'GEE_Wheat_GPC_Monthly';

var MAX_CLOUDY_PIXEL_PCT = 30;
var SCL_BAD = [0, 1, 2, 3, 8, 9, 10, 11];
var SCL_BUFFER = 100;
var EPS = 1e-6;

var S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'SCL'];
var S2_REFL_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];

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

print('Polygons:', polys.size());
print('Months:', MONTHS.length);

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

function extractMonth(feature, year, month) {
  var geom = feature.geometry();
  var start = ee.Date.fromYMD(year, month, 1);
  var end = start.advance(1, 'month');

  var indices = ['NDVI', 'EVI2', 'GNDVI', 'NDRE', 'CIre', 'IRECI', 'NDWI', 'MSI', 'GCVI',
                 'MNDWI', 'NBR', 'NBR2', 'GVI', 'DGCI', 'NDSWIR1RedEdge1', 'TCARI', 'OSAVI',
                 'CCCI', 'mARI', 'ExG', 'DBSI', 'SMMI', 'MIRBI', 'CVI', 'S2REP', 'PVI', 'MTCI',
                 'SAVI', 'NDRE2', 'RE_ratio', 'PSRI', 'LSWI'];
  var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
  var allBands = indices.concat(bands);

  // S2 composites
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUDY_PIXEL_PCT))
    .select(S2_BANDS)
    .map(maskClouds)
    .map(function(img) { return addIndices(toReflectance(img)); });

  var s2Count = s2.size();

  // NaN for months with no valid imagery
  var emptyDict = ee.Dictionary.fromLists(allBands, ee.List.repeat(ee.Number(0).divide(0), allBands.length));

  var s2Stats = ee.Algorithms.If(
    s2Count.gt(0),
    s2.mean().select(allBands).reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: geom,
      scale: SCALE,
      tileScale: TILE_SCALE,
      maxPixels: 1e9,
      bestEffort: true
    }),
    emptyDict
  );
  s2Stats = ee.Dictionary(s2Stats);

  var validPixels = ee.Algorithms.If(
    s2Count.gt(0),
    s2.select('NDVI').count().reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: geom,
      scale: SCALE,
      maxPixels: 1e8
    }).get('NDVI'),
    0
  );

  // ERA5-Land meteo
  var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(start, end);

  var meteo = ee.Image.cat([
    era5.select('temperature_2m').mean().subtract(273.15).rename('t2m_mean'),
    era5.select('temperature_2m_min').min().subtract(273.15).rename('t2m_min'),
    era5.select('temperature_2m_max').max().subtract(273.15).rename('t2m_max'),
    era5.select('total_precipitation_sum').sum().multiply(1000).rename('precip'),
    era5.select('potential_evaporation_sum').sum().multiply(1000).abs().rename('pet'),
    era5.select('temperature_2m').map(function(img) {
      return img.subtract(273.15).max(0);
    }).sum().rename('gdd'),
    era5.select('temperature_2m_max').map(function(img) {
      return img.subtract(273.15).gte(30).selfMask();
    }).count().unmask(0).rename('hot_days'),
    era5.select('dewpoint_temperature_2m').mean().subtract(273.15).rename('dewpoint')
  ]);

  var meteoStats = meteo.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: SCALE,
    tileScale: TILE_SCALE,
    maxPixels: 1e9,
    bestEffort: true
  });

  var allStats = s2Stats.combine(meteoStats);
  return feature.set(allStats).set('s2_count', s2Count).set('valid_pixels', validPixels);
}

// Export monthly CSVs
MONTHS.forEach(function(m) {
  var results = polys.map(function(f) {
    return extractMonth(f, m.year, m.month);
  });

  Export.table.toDrive({
    collection: results,
    description: 'wheat_' + m.name,
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'wheat_' + m.name,
    fileFormat: 'CSV'
  });
});

// Mean elevation per polygon (from SRTM)
var elevResults = polys.map(function(f) {
  var elev = ee.Image('USGS/SRTMGL1_003').select('elevation');
  var stats = elev.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: f.geometry(),
    scale: SCALE,
    maxPixels: 1e8
  });
  return f.set('elevation', stats.get('elevation'));
});

Export.table.toDrive({
  collection: elevResults,
  description: 'wheat_elevation',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'wheat_elevation',
  fileFormat: 'CSV'
});

Map.centerObject(polys, 8);
Map.addLayer(polys, {color: 'green'}, 'Fields');
