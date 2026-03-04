import './style.css'

// True -> test mode | False -> pi data collection
const SIMULATE = false;
const PI_URL = `${window.location.origin}/api/status`; 

const MAPPING = {
  s: 'blue',
  h: 'yellow',
  n: 'white'
};

document.querySelector('#app').innerHTML = `
  <div class="container">
    <div id="box--1" class="box">LEFT</div>
    <div id="box-0" class="box">CENTER</div>
    <div id="box-1" class="box">RIGHT</div>
  </div>
`;

let step = 0;
const testData = [
  { sound: 's', direction: -1 },
  { sound: 'h', direction: 0 },
  { sound: 'n', direction: 1 },
  { sound: 's', direction: 1 },
  { sound: 'h', direction: -1 }
];

async function updateDashboard() {
  let data;

  if (SIMULATE) {
    data = testData[step];
    step = (step + 1) % testData.length;
  } else {
    try {
      const res = await fetch(PI_URL);
      data = await res.json();
    } catch (err) {
      return;
    }
  }

  document.querySelectorAll('.box').forEach(el => {
    el.style.backgroundColor = 'white';
    el.classList.remove('is-active');
  });

  const activeBox = document.getElementById(`box-${data.direction}`);
  if (activeBox) {
    const color = MAPPING[data.sound] || 'white';
    activeBox.style.backgroundColor = color;
    
    if (color !== 'white') {
        activeBox.classList.add('is-active');
    }
  }
}

setInterval(updateDashboard, 1000);
