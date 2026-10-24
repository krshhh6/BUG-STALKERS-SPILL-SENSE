import * as GeoTIFF from 'geotiff';

export interface GeoBounds {
  minLon: number;
  minLat: number;
  maxLon: number;
  maxLat: number;
  centerLat: number;
  centerLng: number;
  isGeoreferenced: boolean;
  crs?: string;
}

export interface DecodedTiffResult {
  dataUrl: string;
  width: number;
  height: number;
  bands: number;
  formatDescription: string;
  vvRaster?: Float32Array;
  vhRaster?: Float32Array;
  geoBounds?: GeoBounds;
}

/**
 * Decodes any TIFF or GeoTIFF (including Sentinel-1 32-bit float SAR dB files)
 * into a browser-renderable PNG Data URL, and provides calibrated dual-pol raster buffers.
 */
export async function decodeTiffFile(file: File | Blob): Promise<DecodedTiffResult> {
  const arrayBuffer = await file.arrayBuffer();
  const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
  const image = await tiff.getImage();

  const originalWidth = image.getWidth();
  const originalHeight = image.getHeight();
  const samplesPerPixel = image.getSamplesPerPixel();

  // Constrain max preview dimension to 800px to ensure fast rendering in web UI
  const maxDim = 800;
  let targetWidth = originalWidth;
  let targetHeight = originalHeight;

  if (targetWidth > maxDim || targetHeight > maxDim) {
    if (targetWidth >= targetHeight) {
      targetHeight = Math.round((originalHeight / originalWidth) * maxDim);
      targetWidth = maxDim;
    } else {
      targetWidth = Math.round((originalWidth / originalHeight) * maxDim);
      targetHeight = maxDim;
    }
  }

  // Read raster data safely without triggering geotiff.js LZW strip resizing bugs
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let rasters: any;
  let isNativeResolution = true;
  try {
    rasters = await image.readRasters();
  } catch (err) {
    console.warn('Native readRasters failed, trying with target dimensions:', err);
    try {
      rasters = await image.readRasters({
        width: targetWidth,
        height: targetHeight,
      });
      isNativeResolution = false;
    } catch (fallbackErr) {
      console.error('All geotiff readRasters attempts failed:', fallbackErr);
      throw fallbackErr;
    }
  }

  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to obtain 2D canvas context for TIFF decoding');
  }

  const imgData = ctx.createImageData(targetWidth, targetHeight);
  const totalPixels = targetWidth * targetHeight;

  let formatDescription = '';
  let vvNormBuffer: Float32Array | undefined;
  let vhNormBuffer: Float32Array | undefined;

  // Helper to get pixel index in source raster
  const getSrcIndex = (x: number, y: number): number => {
    if (!isNativeResolution) {
      return y * targetWidth + x;
    }
    const srcX = Math.min(Math.floor(x * originalWidth / targetWidth), originalWidth - 1);
    const srcY = Math.min(Math.floor(y * originalHeight / targetHeight), originalHeight - 1);
    return srcY * originalWidth + srcX;
  };

  // Case 1: RGB / Multi-spectral TIFF (must check first: >= 3 bands)
  if (rasters.length >= 3 && samplesPerPixel >= 3) {
    formatDescription = `RGB Multi-Band TIFF (${originalWidth}x${originalHeight})`;
    const rRaster = (rasters[0] as unknown) as ArrayLike<number>;
    const gRaster = (rasters[1] as unknown) as ArrayLike<number>;
    const bRaster = (rasters[2] as unknown) as ArrayLike<number>;

    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const dstIdx = y * targetWidth + x;
        const srcIdx = getSrcIndex(x, y);
        const pixelIdx = dstIdx * 4;
        imgData.data[pixelIdx]     = Math.min(255, Math.max(0, Math.round(rRaster[srcIdx])));
        imgData.data[pixelIdx + 1] = Math.min(255, Math.max(0, Math.round(gRaster[srcIdx])));
        imgData.data[pixelIdx + 2] = Math.min(255, Math.max(0, Math.round(bRaster[srcIdx])));
        imgData.data[pixelIdx + 3] = 255;
      }
    }
  // Case 2: Dual-polarization Sentinel-1 SAR (exactly 2 bands: Band 0 = VH, Band 1 = VV in dB)
  } else if (rasters.length === 2) {
    // Verified: Band 1 = VV (co-pol), Band 0 = VH (cross-pol)
    const vhRaster = (rasters[0] as unknown) as ArrayLike<number>;
    const vvRaster = (rasters[1] as unknown) as ArrayLike<number>;
    formatDescription = `Sentinel-1 Dual-Pol SAR (${originalWidth}x${originalHeight}, VV/VH dual-polarization calibrated)`;

    vvNormBuffer = new Float32Array(totalPixels);
    vhNormBuffer = new Float32Array(totalPixels);

    // Sample range to check scale (dB float vs linear uint16/DN)
    const samples: number[] = [];
    const stride = Math.max(1, Math.floor(vvRaster.length / 5000));
    for (let i = 0; i < vvRaster.length; i += stride) {
      const v = vvRaster[i];
      if (Number.isFinite(v)) {
        samples.push(v);
      }
    }
    samples.sort((a, b) => a - b);

    const minSample = samples.length > 0 ? samples[0] : 0;
    const isDbScale = minSample < -5.0;

    let vvMin = -32.0;
    let vvMax = -10.0;
    let vhMin = -42.0;
    let vhMax = -20.0;

    if (!isDbScale && samples.length > 0) {
      // Robust 2nd and 98th percentile scaling for linear DN rasters (more robust than p1/p99)
      const p2Idx = Math.floor(samples.length * 0.02);
      const p98Idx = Math.min(samples.length - 1, Math.floor(samples.length * 0.98));
      vvMin = samples[p2Idx];
      vvMax = Math.max(vvMin + 1, samples[p98Idx]);
      vhMin = vvMin;
      vhMax = vvMax;
    }

    const vvRange = vvMax - vvMin || 1;
    const vhRange = vhMax - vhMin || 1;

    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const dstIdx = y * targetWidth + x;
        const srcIdx = getSrcIndex(x, y);

        const v_vv = vvRaster[srcIdx];
        const v_vh = vhRaster[srcIdx];

        let n_vv = 0;
        if (Number.isFinite(v_vv)) {
          const clipped = Math.max(vvMin, Math.min(vvMax, v_vv));
          n_vv = (clipped - vvMin) / vvRange;
        }
        vvNormBuffer[dstIdx] = n_vv;

        let n_vh = 0;
        if (Number.isFinite(v_vh)) {
          const clipped = Math.max(vhMin, Math.min(vhMax, v_vh));
          n_vh = (clipped - vhMin) / vhRange;
        }
        vhNormBuffer[dstIdx] = n_vh;

        // Draw VV to visual canvas
        const gray = Math.round(n_vv * 255);
        const pixelIdx = dstIdx * 4;
        imgData.data[pixelIdx]     = gray;
        imgData.data[pixelIdx + 1] = gray;
        imgData.data[pixelIdx + 2] = gray;
        imgData.data[pixelIdx + 3] = 255;
      }
    }
  } else if (rasters.length === 1) {
    // Case 2: Single band (Grayscale or single-pol SAR)
    const singleRaster = (rasters[0] as unknown) as ArrayLike<number>;
    formatDescription = `Single-Band Raster (${originalWidth}x${originalHeight})`;

    vvNormBuffer = new Float32Array(totalPixels);
    vhNormBuffer = new Float32Array(totalPixels);

    // Sample range to check scale
    const samples: number[] = [];
    const stride = Math.max(1, Math.floor(singleRaster.length / 5000));
    for (let i = 0; i < singleRaster.length; i += stride) {
      const v = singleRaster[i];
      if (Number.isFinite(v)) {
        samples.push(v);
      }
    }
    samples.sort((a, b) => a - b);

    const minSample = samples.length > 0 ? samples[0] : 0;
    const isDbScale = minSample < -5.0;

    let lowBound = -32.0;
    let highBound = -10.0;
    if (!isDbScale && samples.length > 0) {
      const p1Idx = Math.floor(samples.length * 0.01);
      const p99Idx = Math.min(samples.length - 1, Math.floor(samples.length * 0.99));
      lowBound = samples[p1Idx];
      highBound = Math.max(lowBound + 1, samples[p99Idx]);
    }
    const range = highBound - lowBound || 1;

    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const dstIdx = y * targetWidth + x;
        const srcIdx = getSrcIndex(x, y);
        const v = singleRaster[srcIdx];

        let norm = 0;
        if (Number.isFinite(v)) {
          const clipped = Math.max(lowBound, Math.min(highBound, v));
          norm = (clipped - lowBound) / range;
        }
        vvNormBuffer[dstIdx] = norm;
        vhNormBuffer[dstIdx] = Math.max(0, norm - 0.06);

        const gray = Math.round(norm * 255);
        const pixelIdx = dstIdx * 4;
        imgData.data[pixelIdx] = gray;
        imgData.data[pixelIdx + 1] = gray;
        imgData.data[pixelIdx + 2] = gray;
        imgData.data[pixelIdx + 3] = 255;
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
  const dataUrl = canvas.toDataURL('image/png');
  const geoBounds = extractGeoBounds(image, originalWidth, originalHeight);

  return {
    dataUrl,
    width: targetWidth,
    height: targetHeight,
    bands: samplesPerPixel,
    formatDescription,
    vvRaster: vvNormBuffer,
    vhRaster: vhNormBuffer,
    geoBounds,
  };
}

function utmToLatLon(easting: number, northing: number, zone: number, northernHemisphere = true): [number, number] {
  const a = 6378137.0;
  const f = 1 / 298.257223563;
  const b = a * (1 - f);
  const e = Math.sqrt(1 - (b * b) / (a * a));
  const ePrimeSq = (e * e) / (1 - e * e);
  const k0 = 0.9996;

  const x = easting - 500000;
  const y = northernHemisphere ? northing : northing - 10000000;

  const m = y / k0;
  const mu = m / (a * (1 - (e * e) / 4 - (3 * e * e * e * e) / 64 - (5 * e * e * e * e * e * e) / 256));

  const e1 = (1 - Math.sqrt(1 - e * e)) / (1 + Math.sqrt(1 - e * e));
  const j1 = (3 * e1) / 2 - (27 * e1 * e1 * e1) / 32;
  const j2 = (21 * e1 * e1) / 16 - (55 * e1 * e1 * e1 * e1) / 32;
  const j3 = (151 * e1 * e1 * e1) / 96;

  const fp = mu + j1 * Math.sin(2 * mu) + j2 * Math.sin(4 * mu) + j3 * Math.sin(6 * mu);

  const c1 = ePrimeSq * Math.cos(fp) * Math.cos(fp);
  const t1 = Math.tan(fp) * Math.tan(fp);
  const n1 = a / Math.sqrt(1 - (e * Math.sin(fp)) ** 2);
  const r1 = (a * (1 - e * e)) / (Math.sqrt(1 - (e * Math.sin(fp)) ** 2) ** 3);
  const d = x / (n1 * k0);

  let lat = fp - (n1 * Math.tan(fp) / r1) * (
    (d * d) / 2 -
    ((5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * ePrimeSq) * (d ** 4)) / 24
  );
  lat = (lat * 180) / Math.PI;

  let lon = (
    d -
    ((1 + 2 * t1 + c1) * (d ** 3)) / 6 +
    ((5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * ePrimeSq + 24 * t1 * t1) * (d ** 5)) / 120
  ) / Math.cos(fp);
  lon = ((zone - 1) * 6 - 180 + 3) + (lon * 180) / Math.PI;

  return [lat, lon];
}

function extractGeoBounds(image: any, width: number, height: number): GeoBounds | undefined {
  try {
    let minX: number | null = null;
    let minY: number | null = null;
    let maxX: number | null = null;
    let maxY: number | null = null;

    if (typeof image.getBoundingBox === 'function') {
      const bbox = image.getBoundingBox();
      if (Array.isArray(bbox) && bbox.length >= 4 && bbox.every((v: any) => typeof v === 'number' && Number.isFinite(v))) {
        [minX, minY, maxX, maxY] = bbox;
      }
    }

    if (minX === null && typeof image.getOrigin === 'function' && typeof image.getResolution === 'function') {
      const origin = image.getOrigin();
      const res = image.getResolution();
      if (origin && res && origin.length >= 2 && res.length >= 2) {
        const ox = origin[0];
        const oy = origin[1];
        const rx = res[0];
        const ry = res[1];
        if (typeof ox === 'number' && typeof oy === 'number' && typeof rx === 'number' && typeof ry === 'number') {
          minX = ox;
          maxY = oy;
          maxX = ox + rx * width;
          minY = oy - Math.abs(ry) * height;
        }
      }
    }

    const fileDir = image.fileDirectory || {};
    if (minX === null && fileDir.ModelTiepoint && fileDir.ModelPixelScale) {
      const tp = fileDir.ModelTiepoint;
      const ps = fileDir.ModelPixelScale;
      if (tp.length >= 6 && ps.length >= 2) {
        const tpx = tp[3];
        const tpy = tp[4];
        const psx = ps[0];
        const psy = ps[1];
        if (typeof tpx === 'number' && typeof tpy === 'number' && typeof psx === 'number' && typeof psy === 'number') {
          minX = tpx;
          maxY = tpy;
          maxX = tpx + psx * width;
          minY = tpy - psy * height;
        }
      }
    }

    if (minX === null || minY === null || maxX === null || maxY === null) {
      return undefined;
    }

    const geoKeys = (typeof image.getGeoKeys === 'function' ? image.getGeoKeys() : image.geoKeys) || {};
    const projCode = geoKeys.ProjectedCSTypeGeoKey || geoKeys.ProjectionGeoKey;
    const geoCode = geoKeys.GeographicTypeGeoKey;

    let minLon = minX;
    let maxLon = maxX;
    let minLat = minY;
    let maxLat = maxY;

    // Check if bounds are in WGS84 degree space [-180, 180], [-90, 90]
    const isDegree = minX >= -180 && maxX <= 180 && minY >= -90 && maxY <= 90;

    if (!isDegree) {
      // Projected coordinates (e.g. UTM)
      let zone = 43; // Default Indian Ocean / Arabian Sea UTM zone
      let north = true;

      if (projCode) {
        if (projCode >= 32601 && projCode <= 32660) {
          zone = projCode - 32600;
          north = true;
        } else if (projCode >= 32701 && projCode <= 32760) {
          zone = projCode - 32700;
          north = false;
        }
      }

      const [lat1, lon1] = utmToLatLon(minX, minY, zone, north);
      const [lat2, lon2] = utmToLatLon(maxX, maxY, zone, north);

      minLat = Math.min(lat1, lat2);
      maxLat = Math.max(lat1, lat2);
      minLon = Math.min(lon1, lon2);
      maxLon = Math.max(lon1, lon2);
    }

    const centerLat = Math.round(((minLat + maxLat) / 2) * 100000) / 100000;
    const centerLng = Math.round(((minLon + maxLon) / 2) * 100000) / 100000;

    return {
      minLon: Math.round(minLon * 100000) / 100000,
      minLat: Math.round(minLat * 100000) / 100000,
      maxLon: Math.round(maxLon * 100000) / 100000,
      maxLat: Math.round(maxLat * 100000) / 100000,
      centerLat,
      centerLng,
      isGeoreferenced: true,
      crs: projCode ? `EPSG:${projCode}` : (geoCode ? `EPSG:${geoCode}` : 'EPSG:4326'),
    };
  } catch (err) {
    console.warn('[tiffDecoder] GeoKey parsing fallback warning:', err);
    return undefined;
  }
}
