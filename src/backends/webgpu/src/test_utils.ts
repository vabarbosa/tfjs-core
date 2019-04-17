import {ALL_ENVS, describeWithFlags, registerTestEnv} from '@tensorflow/tfjs-core/src/jasmine_util';

import {ready} from './index';

registerTestEnv({name: 'webgpu', backendName: 'cpu'});

export function describeWebGPU(testName: string, tests: () => void) {
  describeWithFlags(testName, ALL_ENVS, () => {
    console.log('RUNNING TESTS');
    beforeAll(async () => {
      console.log('IN BEFORE ALL');
      return await ready;
    });
    tests();
  });
}