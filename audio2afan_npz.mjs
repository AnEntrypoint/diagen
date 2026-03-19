import { createInflate, inflateSync } from 'zlib';
import { gunzipSync } from 'zlib';

const NPZ_MAGIC = new Uint8Array([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]);

function readNpyHeader(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const magic = buffer.slice(0, 6);
  let offset = 6;
  
  const major = view.getUint8(offset++);
  const minor = view.getUint8(offset++);
  
  let headerLen;
  if (major >= 2) {
    headerLen = view.getUint32(offset, true);
    offset += 4;
  } else {
    headerLen = view.getUint16(offset, true);
    offset += 2;
  }
  
  const header = new TextDecoder().decode(buffer.slice(offset, offset + headerLen));
  const match = header.match(/'descr':\s*'([<>|])([a-z])(\d+)'/i);
  if (!match) throw new Error('Cannot parse numpy dtype');
  
  const [, endian, type, sizeStr] = match;
  const size = parseInt(sizeStr, 10);
  const littleEndian = endian !== '>';
  
  let TypedArray;
  switch (type.toLowerCase()) {
    case 'f': TypedArray = size === 4 ? Float32Array : Float64Array; break;
    case 'i': TypedArray = size === 4 ? Int32Array : size === 2 ? Int16Array : Int8Array; break;
    case 'u': TypedArray = size === 4 ? Uint32Array : size === 2 ? Uint16Array : Uint8Array; break;
    default: TypedArray = Float32Array;
  }
  
  const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/);
  const shape = shapeMatch 
    ? shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
    : [];
  
  return { TypedArray, littleEndian, shape, headerEnd: offset + headerLen };
}

function parseNpy(buffer) {
  const { TypedArray, littleEndian, shape, headerEnd } = readNpyHeader(buffer);
  const data = buffer.slice(headerEnd);
  const arr = new TypedArray(data.buffer, data.byteOffset, data.byteLength / TypedArray.BYTES_PER_ELEMENT);
  return { data: arr, shape };
}

async function loadNpz(filePath) {
  const fs = await import('fs');
  const { default: JSZip } = await import('jszip');
  
  const buffer = fs.readFileSync(filePath);
  const zip = await JSZip.loadAsync(buffer);
  
  const result = {};
  for (const [name, file] of Object.entries(zip.files)) {
    if (name.endsWith('.npy')) {
      const npyBuffer = await file.async('uint8array');
      const { data, shape } = parseNpy(npyBuffer);
      const key = name.replace('.npy', '');
      result[key] = data;
      result[key].shape = shape;
    }
  }
  return result;
}

export { loadNpz, parseNpy };
