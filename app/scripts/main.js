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
const result = document.getElementById('result');
const resultContent = result.lastElementChild;

window.openResult = function (data) {
	resultContent.innerHTML = `
        <i data-lucide="${data.icon}"></i>
        <h2>${data.name}</h2>
		<div class="result-box">
			<h3>Details</h3>
            <div>
				<h4>Material(s):</h4>
				<span>${data.material}</span>
			</div>
            <div>
				<h4>Disposal method:</h4>
				<span>${data.method}</span>
			</div>
        </div>
        <div class="result-box">
			<h3>Environmental impact</h3>
            <div>
				<h4>Carbon footprint:</h4>
				<span style="color: ${data.footprint < 0.2 ? '#22c55e'
              : data.footprint < 0.4 ? '#fb923c' 
              : 'red'}">${data.footprint} kg CO<sub>2</sub>e</span>
			</div>
            <div>
				<h4>Decomposition time:</h4>
				<span style="color: ${data.decomposition <= 5 ? '#22c55e'
              : data.decomposition <= 100 ? '#fb923c' 
              : '#ef4444'}">${data.decomposition.toLocaleString()} year${data.decomposition > 1 ? 's' : ''}</span>
			</div>
        </div>
	`;
	updateIcons();

	mlStop();
	setTimeout(() => {
		result.style.left = '0%';
	}, 100);
};

document.getElementById('result-close').addEventListener('click', () => {
	result.style.left = '100%';
	// if menu is not open, resume ml
	if (menu.style.left != '0%') {
		setTimeout(() => {
			mlResume();
		}, 500);
	}
});

// initialization
updateIcons();
getCamera(facingMode);
