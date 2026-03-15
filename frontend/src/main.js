/* SCOPE - Fraud Detection Simulator
   Main application logic */

const API_BASE = 'http://localhost:8000';

// -- DOM refs --
const form = document.getElementById('purchase-form');
const submitBtn = document.getElementById('submit-btn');
const btnText = submitBtn.querySelector('.btn-text');
const btnLoader = submitBtn.querySelector('.btn-loader');
const resultsCard = document.getElementById('results-card');
const timestampInput = document.getElementById('timestamp');
const phoneLatInput = document.getElementById('phone-lat');
const phoneLngInput = document.getElementById('phone-lng');
const msgBadge = document.getElementById('msg-badge');
const messageThread = document.getElementById('messages-thread');
const messageInput = document.getElementById('message-input');
const messageSend = document.getElementById('message-send');
const mapCoordsLabel = document.getElementById('map-coords');
const btnSetLocation = document.getElementById('btn-set-location');
const statusTime = document.getElementById('status-time');

// -- State --
let selectedPhoneLat = null;
let selectedPhoneLng = null;
let leafletMap = null;
let mapMarker = null;
let heatmapMap = null;
let heatmapLayer = null; // heatmap.js overlay
let heatmapLayers = [];
let pendingTxn = null; // stores last high/med-risk transaction for 2FA
let lastTxnLat = null;
let lastTxnLng = null;

// STATUS BAR CLOCK
function updateClock() {
  const now = new Date();
  statusTime.textContent = now.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: false
  });
}
updateClock();
setInterval(updateClock, 30000);

// SCREEN NAVIGATION
const screens = document.querySelectorAll('.screen');

function showScreen(screenId) {
  screens.forEach(s => {
    if (s.classList.contains('active')) {
      s.classList.remove('active');
      s.classList.add('slide-out');
    } else {
      s.classList.remove('slide-out');
    }
  });
  const target = document.getElementById(`screen-${screenId}`);
  if (target) {
    // Delay slightly so the outgoing transition starts first
    requestAnimationFrame(() => {
      target.classList.remove('slide-out');
      target.classList.add('active');
    });
  }
  // Initialize map when Maps screen opens
  if (screenId === 'maps' && !leafletMap) {
    setTimeout(initMap, 100);
  }
  // Initialize heatmap when Heatmap screen opens
  if (screenId === 'heatmap') {
    setTimeout(() => {
      initHeatmapMap();
      loadHeatmap();
    }, 100);
  }
  // Load transactions when transactions screen opens
  if (screenId === 'transactions') {
    loadTransactions();
  }
}

// App icon clicks
document.querySelectorAll('[data-screen]').forEach(btn => {
  btn.addEventListener('click', () => {
    const screen = btn.dataset.screen;
    if (screen) showScreen(screen);
  });
});

// LEAFLET MAP (Phone Location)
function initMap() {
  const container = document.getElementById('leaflet-map');
  if (!container || leafletMap) return;

  leafletMap = L.map(container, {
    center: [40.7128, -74.0060],
    zoom: 12,
    zoomControl: true,
    attributionControl: true,
  });

  // Dark-ish tile layer (CartoDB dark matter)
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '(c)OpenStreetMap (c)CartoDB',
    maxZoom: 19,
    subdomains: 'abcd',
  }).addTo(leafletMap);

  // Click to place marker
  leafletMap.on('click', (e) => {
    const { lat, lng } = e.latlng;
    selectedPhoneLat = Math.round(lat * 1e6) / 1e6;
    selectedPhoneLng = Math.round(lng * 1e6) / 1e6;

    if (mapMarker) {
      mapMarker.setLatLng(e.latlng);
    } else {
      mapMarker = L.marker(e.latlng, {
        icon: L.divIcon({
          className: '',
          html: `<div style="width:24px;height:24px;background:#2D82B7;border:3px solid #F3DFBF;border-radius:50%;box-shadow:0 2px 8px rgba(0,0,0,0.4);"></div>`,
          iconSize: [24, 24],
          iconAnchor: [12, 12],
        }),
      }).addTo(leafletMap);
    }

    mapCoordsLabel.textContent = `${selectedPhoneLat}, ${selectedPhoneLng}`;
    btnSetLocation.disabled = false;
  });

  // Force map to resize properly
  setTimeout(() => leafletMap.invalidateSize(), 200);
}

// Set Location button
btnSetLocation.addEventListener('click', () => {
  if (selectedPhoneLat == null) return;

  phoneLatInput.value = selectedPhoneLat;
  phoneLngInput.value = selectedPhoneLng;

  // Flash green on the form inputs
  [phoneLatInput, phoneLngInput].forEach(el => {
    el.classList.add('phone-lat-set');
    setTimeout(() => el.classList.remove('phone-lat-set'), 700);
  });

  // Go back to home
  showScreen('home');
});

// HEATMAP MAP
function initHeatmapMap() {
  const container = document.getElementById('heatmap-map');
  if (!container || heatmapMap) {
    if (heatmapMap) setTimeout(() => heatmapMap.invalidateSize(), 100);
    return;
  }

  heatmapMap = L.map(container, {
    center: [38, -82],
    zoom: 5,
    zoomControl: true,
    attributionControl: true,
  });

  // Initialize heatmap layer
  const cfg = {
    "radius": 40,
    "maxOpacity": .6,
    "scaleRadius": false,
    "useLocalExtrema": true,
    latField: 'lat',
    lngField: 'lng',
    valueField: 'count'
  };

  heatmapLayer = new HeatmapOverlay(cfg);
  heatmapLayer.addTo(heatmapMap);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '(c)OpenStreetMap (c)CartoDB',
    maxZoom: 19,
    subdomains: 'abcd',
  }).addTo(heatmapMap);

  setTimeout(() => heatmapMap.invalidateSize(), 200);
}

async function loadHeatmap() {
  if (!heatmapMap || !heatmapLayer) return;

  // Clear previous non-heatmap layers
  heatmapLayers.forEach(layer => heatmapMap.removeLayer(layer));
  heatmapLayers = [];
  heatmapLayer.setData({ max: 0, data: [] });

  const userId = document.getElementById('user-id').value;

  try {
    // 1. Load User Profile for zones
    const profileRes = await fetch(`${API_BASE}/user-profile/${userId}`);
    let zones = [];
    let bounds = L.latLngBounds();

    if (profileRes.ok) {
      const profile = await profileRes.json();
      zones = profile.frequent_zones || [];
      
      zones.forEach((zone, i) => {
        // Draw zone radius circle
        const circle = L.circle([zone.lat, zone.lng], {
          radius: zone.radius_km * 1000,
          color: 'var(--accent)',
          fillColor: 'var(--accent)',
          fillOpacity: 0.08,
          weight: 1,
          dashArray: '4 4',
        }).addTo(heatmapMap);
        heatmapLayers.push(circle);
        bounds.extend(circle.getBounds());

        // Zone center marker
        const centerMarker = L.circleMarker([zone.lat, zone.lng], {
          radius: 4,
          color: 'var(--accent)',
          fillColor: 'var(--accent)',
          fillOpacity: 0.6,
          weight: 1,
        }).addTo(heatmapMap);
        heatmapLayers.push(centerMarker);
      });
    }

    // 2. Build Heatmap Data
    let heatmapPoints = [];

    // Add points for zones (to show they are 'trusted' hotspots)
    zones.forEach(zone => {
      // Create a cluster of points to simulate high density in the trusted zone
      for (let i = 0; i < 15; i++) {
        // Random offset within radius
        const angle = Math.random() * Math.PI * 2;
        const dist = Math.random() * (zone.radius_km / 111.12); // rough deg to km conversion
        heatmapPoints.push({
          lat: zone.lat + Math.cos(angle) * dist,
          lng: zone.lng + Math.sin(angle) * dist,
          count: 0.8
        });
      }
    });

    // 3. Load Transactions for Heatmap
    const txnRes = await fetch(`${API_BASE}/transactions/${userId}`);
    if (txnRes.ok) {
      const data = await txnRes.json();
      const txns = data.transactions || [];
      
      if (txns.length > 0) {
        txns.forEach(log => {
          heatmapPoints.push({
            lat: log.transaction.transaction_lat,
            lng: log.transaction.transaction_lng,
            count: 1
          });
          bounds.extend([log.transaction.transaction_lat, log.transaction.transaction_lng]);
        });
      }
    }

    // Set combined data
    heatmapLayer.setData({
      max: 5,
      data: heatmapPoints
    });

    // 4. Show last transaction marker
    if (lastTxnLat != null && lastTxnLng != null) {
      const txnMarker = L.circleMarker([lastTxnLat, lastTxnLng], {
        radius: 8,
        color: '#EB8A90',
        fillColor: '#EB8A90',
        fillOpacity: 1,
        weight: 3,
        className: 'pulse-marker'
      }).addTo(heatmapMap);
      txnMarker.bindPopup('Last Transaction');
      heatmapLayers.push(txnMarker);
      bounds.extend([lastTxnLat, lastTxnLng]);
    }

    if (bounds.isValid()) {
      heatmapMap.fitBounds(bounds, { padding: [40, 40] });
    }
  } catch (err) {
    console.error('Failed to load heatmap:', err);
  }
}

// View Heatmap button on left panel
document.getElementById('btn-view-heatmap').addEventListener('click', () => {
  showScreen('heatmap');
});

// FORM SUBMISSION
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  // Auto-fill timestamp
  timestampInput.value = new Date().toISOString();

  // Validate phone location
  if (!phoneLatInput.value || !phoneLngInput.value) {
    alert('Please set your phone location using the Maps app on the iPhone first!');
    return;
  }

  const payload = {
    user_id: document.getElementById('user-id').value,
    transaction_id: `txn_${Date.now()}`,
    amount: parseFloat(document.getElementById('amount').value),
    merchant_name: document.getElementById('merchant-name').value,
    merchant_category: document.getElementById('merchant-category').value,
    transaction_lat: parseFloat(document.getElementById('tx-lat').value),
    transaction_lng: parseFloat(document.getElementById('tx-lng').value),
    phone_lat: parseFloat(phoneLatInput.value),
    phone_lng: parseFloat(phoneLngInput.value),
    timestamp: timestampInput.value,
  };

  // Store transaction location for heatmap
  lastTxnLat = payload.transaction_lat;
  lastTxnLng = payload.transaction_lng;

  // Show loading state
  btnText.hidden = true;
  btnLoader.hidden = false;
  submitBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/score-transaction`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Server error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    displayResults(data);

    // If risky, trigger 2FA
    if (data.label === 'HIGH_RISK' || data.label === 'MEDIUM_RISK') {
      pendingTxn = { ...payload, result: data };
      trigger2FA(payload, data);
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btnText.hidden = false;
    btnLoader.hidden = true;
    submitBtn.disabled = false;
  }
});

// DISPLAY RESULTS
function displayResults(data) {
  resultsCard.hidden = false;

  const badge = document.getElementById('risk-badge');
  badge.textContent = data.label.replace('_', ' ');
  badge.className = 'risk-badge';
  if (data.label === 'LOW_RISK') badge.classList.add('low');
  else if (data.label === 'MEDIUM_RISK') badge.classList.add('medium');
  else badge.classList.add('high');

  document.getElementById('fraud-prob').textContent = (data.fraud_probability * 100).toFixed(1) + '%';
  document.getElementById('ml-score').textContent = (data.ml_score * 100).toFixed(1) + '%';
  document.getElementById('heatmap-score').textContent = (data.heatmap_score * 100).toFixed(1) + '%';
  document.getElementById('final-score').textContent = (data.final_score * 100).toFixed(1) + '%';

  const reasonsList = document.getElementById('reasons-list');
  reasonsList.innerHTML = '';
  data.reasons.forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    reasonsList.appendChild(li);
  });

  // Scroll results into view
  resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 2FA MESSAGES
function trigger2FA(txn, result) {
  // Show notification badge
  msgBadge.hidden = false;

  // Add 2FA message
  addMessage(
    'incoming',
    `SCOPE Security: A purchase of $${txn.amount.toFixed(2)} at "${txn.merchant_name}" was flagged as ${result.label.replace('_', ' ')}. Risk score: ${(result.final_score * 100).toFixed(0)}%.\n\nReply YES to approve or NO to decline.`
  );

  // Enable input
  messageInput.disabled = false;
  messageSend.disabled = false;
  messageInput.placeholder = 'Type YES or NO...';

  // Switch to messages after a short delay
  setTimeout(() => showScreen('messages'), 800);
}

function addMessage(type, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${type}`;
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text;
  wrapper.appendChild(bubble);
  messageThread.appendChild(wrapper);
  messageThread.scrollTop = messageThread.scrollHeight;
}

// Send message handler
function handleSend() {
  const text = messageInput.value.trim().toUpperCase();
  if (!text) return;

  addMessage('outgoing', messageInput.value.trim());
  messageInput.value = '';

  if (text === 'YES') {
    setTimeout(() => {
      addMessage('incoming', 'Transaction approved. Thank you for confirming.');
      messageInput.disabled = true;
      messageSend.disabled = true;
      msgBadge.hidden = true;
      pendingTxn = null;
    }, 500);
  } else if (text === 'NO') {
    setTimeout(() => {
      addMessage('incoming', 'Transaction declined. Your card has been temporarily locked. Contact support if this was a mistake.');
      messageInput.disabled = true;
      messageSend.disabled = true;
      msgBadge.hidden = true;
      pendingTxn = null;
    }, 500);
  } else {
    setTimeout(() => {
      addMessage('incoming', 'Please reply with YES to approve or NO to decline the transaction.');
    }, 400);
  }
}

messageSend.addEventListener('click', handleSend);
messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') handleSend();
});

// TRANSACTIONS LIST
async function loadTransactions() {
  const userId = document.getElementById('user-id').value;
  const listContainer = document.getElementById('transactions-list');
  const emptyState = document.getElementById('transactions-empty');

  listContainer.innerHTML = '';
  emptyState.hidden = true;

  try {
    const res = await fetch(`${API_BASE}/transactions/${userId}`);
    if (!res.ok) throw new Error('Failed to fetch transactions');
    const data = await res.json();
    const txns = data.transactions || [];

    if (txns.length === 0) {
      emptyState.hidden = false;
      return;
    }

    // Sort by created_at descending
    txns.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    txns.forEach(log => {
      const txn = log.transaction;
      const decision = log.decision;
      const date = new Date(log.created_at);

      const item = document.createElement('div');
      item.className = 'transaction-item';

      const statusClass = decision.label.toLowerCase().replace('_risk', '');
      const displayStatus = decision.label.replace('_', ' ');

      item.innerHTML = `
        <div class="txn-left">
          <span class="txn-merchant">${txn.merchant_name}</span>
          <span class="txn-category">${txn.merchant_category.replace('_', ' ')}</span>
          <span class="txn-date">${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div class="txn-right">
          <span class="txn-amount">$${txn.amount.toFixed(2)}</span>
          <span class="txn-status-badge ${statusClass}">${displayStatus}</span>
        </div>
      `;
      listContainer.appendChild(item);
    });
  } catch (err) {
    console.error('Error loading transactions:', err);
    emptyState.innerHTML = '<p style="color:var(--red)">Failed to load transactions</p>';
    emptyState.hidden = false;
  }
}

// INIT
// Auto-fill timestamp on load
timestampInput.value = new Date().toISOString();

