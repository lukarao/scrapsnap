import { processFrame } from './ml.js';

import {
	createIcons,
	Flashlight,
	FlashlightOff,
	RefreshCw,
	Menu,
} from 'https://cdn.jsdelivr.net/npm/lucide/+esm';

const icons = {
	icons: {
		Flashlight,
		FlashlightOff,
		RefreshCw,
		Menu,
	},
};

// camera functionality
const video = document.getElementById('video');
let track;

async function getCamera(facingMode) {
	const mediaStream = await navigator.mediaDevices.getUserMedia({
		video: { facingMode: facingMode },
	});
	video.srcObject = mediaStream;
	video.play();

	// initialize ml frame processing
	video.requestVideoFrameCallback(processFrame);

	track = mediaStream.getVideoTracks()[0];
}

// flashlight button
const flashlight = document.getElementById('flashlight');
let torch = false;

function updateFlashlightButton() {
	if (torch) {
		flashlight.innerHTML =
			'<i data-lucide="flashlight-off" class="button"></i>';
		createIcons(icons);
	} else {
		flashlight.innerHTML = '<i data-lucide="flashlight" class="button"></i>';
		createIcons(icons);
	}
}

document.getElementById('flashlight').addEventListener('click', () => {
	track
		.applyConstraints({
			advanced: [{ torch: !torch }],
		})
		.then(() => {
			torch = !torch;
			updateFlashlightButton();
		});
});

// flip camera button
let facingMode = 'environment';

document.getElementById('flip').addEventListener('click', () => {
	const newFacingMode = facingMode == 'environment' ? 'user' : 'environment';
	getCamera(newFacingMode).then(() => {
		facingMode = newFacingMode;
		torch = false;
		updateFlashlightButton();
	});
});

// menu button
document.getElementById('menu').addEventListener('click', () => {
	// TODO: add menu button functionality
});

// initialization
createIcons(icons);
getCamera(facingMode);
