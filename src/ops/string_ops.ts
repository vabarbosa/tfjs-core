/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import {StringTensor, Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';

import {op} from './operation';

// Running in Node.js
const runningInNode = typeof Buffer !== 'undefined' &&
    (typeof atob === 'undefined' || typeof btoa === 'undefined');

// make url safe by replacing + and /
function toUrlSafe(str: string): string {
  return str.replace(/\+/g, '-').replace(/\//g, '_');
}

// revert url safe replacement of + and /
function fromUrlSafe(str: string): string {
  return str.replace(/-/g, '+').replace(/_/g, '/');
}

// Convert from UTF-16 character to UTF-8 multibyte sequence
// tslint:disable-next-line: max-line-length
// https://developer.mozilla.org/en-US/docs/Web/API/WindowBase64/Base64_encoding_and_decoding#The_Unicode_Problem
function utf16ToUtf8Multibyte(char: string): string {
  // get the Unicode code point from UTF-16
  const cp = char.codePointAt(0);

  // convert Unicode code point to UTF-8
  // https://en.wikipedia.org/wiki/UTF-8#Description
  if (cp < 0x80) {
    // one byte
    return char;
  } else if (cp < 0x800) {
    // two bytes
    return String.fromCharCode(0xc0 | (cp >>> 6)) +
        String.fromCharCode(0x80 | (cp & 0x3f));
  } else if (cp < 0x10000) {
    // three bytes
    return String.fromCharCode(0xe0 | ((cp >>> 12) & 0x0f)) +
        String.fromCharCode(0x80 | ((cp >>> 6) & 0x3f)) +
        String.fromCharCode(0x80 | (cp & 0x3f));
  } else {
    // four bytes
    return (
        String.fromCharCode(0xf0 | ((cp >>> 18) & 0x07)) +
        String.fromCharCode(0x80 | ((cp >>> 12) & 0x3f)) +
        String.fromCharCode(0x80 | ((cp >>> 6) & 0x3f)) +
        String.fromCharCode(0x80 | (cp & 0x3f)));
  }
}

// Convert UTF-8 multibyte sequence to UTF-16 character
// tslint:disable-next-line: max-line-length
// https://developer.mozilla.org/en-US/docs/Web/API/WindowBase64/Base64_encoding_and_decoding#The_Unicode_Problem
function utf8MultibyteToUtf16(seq: string): string {
  let cp = 0;

  // get the Unicode code point from UTF-8 multibyte sequence
  // https://en.wikipedia.org/wiki/UTF-8#Description
  switch (seq.length) {
    // two bytes
    case 2:
      cp = ((0x1f & seq.charCodeAt(0)) << 6) + (0x3f & seq.charCodeAt(1));
      break;
    // three bytes
    case 3:
      cp = ((0x0f & seq.charCodeAt(0)) << 12) +
          ((0x3f & seq.charCodeAt(1)) << 6) + (0x3f & seq.charCodeAt(2));
      break;
    // four bytes
    case 4:
      cp = ((0x07 & seq.charCodeAt(0)) << 18) +
          ((0x3f & seq.charCodeAt(1)) << 12) +
          ((0x3f & seq.charCodeAt(2)) << 6) + (0x3f & seq.charCodeAt(3));
      break;
    // one byte
    default:
      return seq;
  }

  // convert Unicode code point to UTF-16
  return String.fromCodePoint(cp);
}

/**
 * Encodes the values of a `tf.Tensor` (of dtype `string`) to Base64.
 *
 * Given a String tensor, returns a new tensor with the values encoded into
 * web-safe base64 format.
 *
 * Web-safe means that the encoder uses `-` and `_` instead of `+` and `/`:
 *
 * en.wikipedia.org/wiki/Base64
 *
 * ```js
 * const x = tf.tensor1d(['Hello World!'], 'string');
 *
 * x.encodeBase64().print();
 * ```
 * @param str The input `tf.Tensor` of dtype `string` to encode.
 * @param pad Whether to add padding (`=`) to the end of the encoded string.
 */
/** @doc {heading: 'Operations', subheading: 'String'} */
function encodeBase64_<T extends StringTensor>(
    str: StringTensor|Tensor, pad = false): T {
  const $str = convertToTensor(str, 'str', 'encodeBase64', 'string');

  const resultValues = new Array($str.size);
  const values = $str.dataSync();

  for (let i = 0; i < values.length; ++i) {
    let bVal = values[i].toString();
    if (runningInNode) {
      bVal = Buffer.from(bVal).toString('base64');
    } else {
      // convert to UTF8 then encode to base64
      bVal = Array.from(bVal).map(utf16ToUtf8Multibyte).join('');
      bVal = btoa(bVal);
    }

    bVal = toUrlSafe(bVal);
    if (!pad) {
      bVal = bVal.replace(/=/g, '');
    }

    resultValues[i] = bVal;
  }

  return Tensor.make($str.shape, {values: resultValues}, $str.dtype) as T;
}

/**
 * Decodes the values of a `tf.Tensor` (of dtype `string`) from Base64.
 *
 * Given a String tensor of Base64 encoded values, returns a new tensor with the
 * decoded values.
 *
 * en.wikipedia.org/wiki/Base64
 *
 * ```js
 * const y = tf.scalar('YW55dGhpbmcgZWxzZSBmb3IgeW91IGdvaW5nIG9uIGhlcmU_',
 *  'string');
 *
 * y.decodeBase64().print();
 * ```
 * @param str The input `tf.Tensor` of dtype `string` to decode.
 */
/** @doc {heading: 'Operations', subheading: 'String'} */
function decodeBase64_<T extends StringTensor>(str: StringTensor|Tensor): T {
  const $str = convertToTensor(str, 'str', 'decodeBase64', 'string');

  const multibyteSeq =
      // tslint:disable-next-line: max-line-length
      /[\xC0-\xDF][\x80-\xBF]|[\xE0-\xEF][\x80-\xBF]{2}|[\xF0-\xF7][\x80-\xBF]{3}/g;

  const resultValues = new Array($str.size);
  const values = $str.dataSync();

  for (let i = 0; i < values.length; ++i) {
    let sVal = fromUrlSafe(values[i].toString());
    if (runningInNode) {
      sVal = Buffer.from(sVal, 'base64').toString();
    } else {
      // decode from base64 then
      // convert to multibyte sequences back to UTF-16
      sVal = atob(sVal).replace(multibyteSeq, utf8MultibyteToUtf16);
    }

    resultValues[i] = sVal;
  }

  return Tensor.make($str.shape, {values: resultValues}, $str.dtype) as T;
}

export const encodeBase64 = op({encodeBase64_});
export const decodeBase64 = op({decodeBase64_});
