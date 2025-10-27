import data from '../data/data.json' with { type: 'json' };

import {
	env,
	InferenceSession,
	Tensor,
} from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/+esm';
env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const session = await InferenceSession.create(
	'../ml/models/final/v3_nms.onnx',
	{
		executionProviders: ['wasm'],
	}
);

const video = document.getElementById('video');
const ctx = document
	.getElementById('canvas')
	.getContext('2d', { willReadFrequently: true });

// https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html
function imageDataToTensor(image, dims) {
	// get buffer data from image and create R, G, and B arrays.
	var imageBufferData = image.data;
	const [redArray, greenArray, blueArray] = new Array(
		new Array(),
		new Array(),
		new Array()
	);

	// loop through the image buffer and extract the R, G, and B channels
	for (let i = 0; i < imageBufferData.length; i += 4) {
		redArray.push(imageBufferData[i]);
		greenArray.push(imageBufferData[i + 1]);
		blueArray.push(imageBufferData[i + 2]);
		// skip data[i + 3] to filter out the alpha channel
	}

	// concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
	const transposedData = redArray.concat(greenArray).concat(blueArray);

	// convert to float32
	let i,
		l = transposedData.length; // length, we need this for the loop
	// create the Float32Array size 3 * 224 * 224 for these dimensions output
	const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
	for (i = 0; i < l; i++) {
		float32Data[i] = transposedData[i] / 255.0; // convert to float
	}
	// create the tensor object from onnxruntime-web.
	const inputTensor = new Tensor('float32', float32Data, dims);
	return inputTensor;
}

const confidenceThreshold = 0.7;

export function processFrame() {
	// capture video frame and scale/crop to 640x640
	const scaledHeight = video.videoHeight * (640 / video.videoWidth);
	ctx.drawImage(video, 0, (640 - scaledHeight) / 2, 640, scaledHeight);
	const imageData = ctx.getImageData(0, 0, 640, 640);

	// convert image data to tensor
	const inputTensor = imageDataToTensor(imageData, [1, 3, 640, 640]);

	// run inference
	session.run({ images: inputTensor }).then((results) => {
		// detection format: [x, y, w, h, conf, label]

		// get best detection
		let bestDetection = results.output0.cpuData.slice(0, 6);
		for (let i = 6; i < 1800; i += 6) {
			const detection = results.output0.cpuData.slice(i, i + 6);
			if (detection[4] > bestDetection[4]) {
				bestDetection = detection;
			}
		}
		if (bestDetection[4] > confidenceThreshold && bestDetection[5] in data) {
			// TODO: do something with the result
			console.log(data[bestDetection[5]]);
		}
	});

	video.requestVideoFrameCallback(processFrame);
}
