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

export const encodeBase64 = op({encodeBase64_});
