import { updateIcons } from './icons.js';

export function card(target, data, cls) {
	target.insertAdjacentHTML(
		'beforeend',
		`<button class="card ${cls}" onclick="openResult(${JSON.stringify(
			data
		).replaceAll('"', "'")})">
            <div class="card-icon">
                <i data-lucide="${data.icon}"></i>
            </div>
            <div class="card-content">
                <h2>${data.name}</h2>
                <h3>${data.type}</h3>
            </div>
            <div class="card-link">
                <i data-lucide="chevron-right"></i>
            </div>
        </button>`
	);
	updateIcons();
}
