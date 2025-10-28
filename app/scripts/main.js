import data from '../data/data.json' with { type: 'json' };

import { card } from './card.js';
import { updateIcons } from './icons.js';
import { processFrame, mlStop, mlResume } from './ml.js';

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
	flashlight.innerHTML = `<i data-lucide="${
		torch ? 'flashlight-off' : 'flashlight'
	}" class="button"></i>`;
	updateIcons();
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

// menu
const menu = document.getElementById('menu');

document.getElementById('menu-open').addEventListener('click', () => {
	mlStop();
	setTimeout(() => {
		menu.style.left = '0%';
	}, 100);
});

document.getElementById('menu-close').addEventListener('click', () => {
	menu.style.left = '100%';
	setTimeout(() => {
		mlResume();
	}, 500);
});

const menuList = document.getElementById('menu-list');

const sortedValues = Object.values(data).sort((a, b) =>
	a.name.localeCompare(b.name)
);
for (const value of sortedValues) {
	card(menuList, value, 'menu-card');
}

const searchInput = document.getElementById('search-input');
searchInput.addEventListener('keyup', () => {
	for (const card of menuList.children) {
		if (
			card
				.querySelector('h2')
				.textContent.toLowerCase()
				.includes(searchInput.value.toLowerCase())
		) {
			card.style.display = 'flex';
		} else {
			card.style.display = 'none';
		}
	}
});

// result
window.openResult = function (data) {
	// TODO: add result code
	console.log(data);
};

// initialization
updateIcons();
getCamera(facingMode);
