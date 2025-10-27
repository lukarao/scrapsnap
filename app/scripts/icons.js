import {
	createIcons,
	icons as lucideIcons,
} from 'https://cdn.jsdelivr.net/npm/lucide/+esm';
import * as lucideLab from 'https://cdn.jsdelivr.net/npm/@lucide/lab/+esm';

// set icons to all icons in Lucide and Lucide lab
// this is probably not efficient
const icons = {
	icons: { ...lucideIcons },
};
// Lucide lab icons are not capitalized, which messes up createIcons()
for (const key in lucideLab) {
	const capitalizedKey = key.charAt(0).toUpperCase() + key.slice(1);
	icons.icons[capitalizedKey] = lucideLab[key];
}

export function updateIcons() {
	createIcons(icons);
}
