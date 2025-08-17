const API_BASE_URL = "https://weather-project-orjd.onrender.com/";
const OPENWEATHER_API_KEY = "b51cb434e01487ae9e1803a8b9ef73d5";
const KASUNGU_COORDS = {
  lat: -13.028085670616038,
  lon: 33.464763622195804,
};

// Global variables
let currentPredictions = null;
let selectedCrop = null;
let selectedMonth = null;
let currentWeatherData = null;
let forecastData = null;

// Comprehensive crop database for Kasungu District
const cropDatabase = {
  maize: {
    name: { en: "Maize", ch: "Chimanga" },
    emoji: "🌽",
    type: "Staple Crop",
    plantingMonths: [11, 12, 1],
    harvestMonths: [4, 5, 6],
    optimalTemp: { min: 18, max: 30 },
    optimalRainfall: { min: 500, max: 1200 },
    diseases: {
      highHumidity: [
        {
          name: "Gray Leaf Spot",
          risk: "High",
          prevention: "Apply fungicides, ensure good air circulation",
        },
        {
          name: "Northern Corn Leaf Blight",
          risk: "High",
          prevention: "Use resistant varieties, crop rotation",
        },
        {
          name: "Maize Rust",
          risk: "Medium",
          prevention: "Early detection, fungicide application",
        },
      ],
      lowRainfall: [
        {
          name: "Drought Stress",
          risk: "High",
          prevention: "Irrigation, drought-tolerant varieties",
        },
        {
          name: "Fall Armyworm",
          risk: "High",
          prevention: "Regular scouting, biological control",
        },
      ],
      highRainfall: [
        {
          name: "Root Rot",
          risk: "Medium",
          prevention: "Improve drainage, avoid waterlogging",
        },
        {
          name: "Stalk Rot",
          risk: "Medium",
          prevention: "Balanced fertilization, proper plant density",
        },
      ],
    },
    recommendations: {
      planting:
        "Plant when soil temperature reaches 16°C. Use certified seeds and apply basal fertilizer (NPK 23:21:0+4S) at 200kg/ha.",
      care: "Side-dress with urea (46% N) at 6-8 weeks after planting. Control weeds early using pre-emergence herbicides. Monitor for Fall Armyworm.",
      harvest:
        "Harvest when moisture content is 20-25% for storage. Proper drying to 12.5% moisture prevents aflatoxin contamination.",
      irrigation:
        "Requires 500-800mm water throughout growing season. Critical periods: tasseling and grain filling.",
      fertilizer:
        "Basal: 8:18:8 + 4S at 300kg/ha. Top dress: Urea at 150kg/ha at 6 weeks after planting.",
    },
  },
  tobacco: {
    name: { en: "Tobacco", ch: "Fodya" },
    emoji: "🚬",
    type: "Cash Crop",
    plantingMonths: [9, 10, 11],
    harvestMonths: [3, 4, 5],
    optimalTemp: { min: 20, max: 28 },
    optimalRainfall: { min: 800, max: 1000 },
    diseases: {
      highHumidity: [
        {
          name: "Blue Mold",
          risk: "Very High",
          prevention: "Metalaxyl fungicide, avoid overhead irrigation",
        },
        {
          name: "Black Shank",
          risk: "High",
          prevention: "Resistant varieties, soil fumigation",
        },
        {
          name: "Bacterial Wilt",
          risk: "Medium",
          prevention: "Crop rotation, resistant varieties",
        },
      ],
      lowRainfall: [
        {
          name: "Drought Stress",
          risk: "High",
          prevention: "Supplemental irrigation, mulching",
        },
        {
          name: "Thrips",
          risk: "Medium",
          prevention: "Insecticide application, reflective mulch",
        },
      ],
    },
    recommendations: {
      planting:
        "Prepare seedbeds in August. Transplant after 8-10 weeks when seedlings are 15cm tall. Plant spacing: 120cm x 60cm.",
      care: "Regular cultivation and pest control. Apply fertilizer in splits: basal, 3 weeks, and 6 weeks after transplanting.",
      harvest:
        "Harvest leaves when mature (yellow-green color). Start from bottom leaves. Cure properly for 6-8 weeks.",
      irrigation:
        "Requires consistent moisture. Avoid water stress during rapid growth phase.",
      fertilizer:
        "Basal: 8:14:6 + 4S at 400kg/ha. Top dress: CAN at 200kg/ha in splits.",
    },
  },
  beans: {
    name: { en: "Common Beans", ch: "Nyemba" },
    emoji: "🫘",
    type: "Legume Crop",
    plantingMonths: [11, 12, 1, 2],
    harvestMonths: [3, 4, 5],
    optimalTemp: { min: 18, max: 25 },
    optimalRainfall: { min: 400, max: 600 },
    diseases: {
      highHumidity: [
        {
          name: "Anthracnose",
          risk: "High",
          prevention: "Copper-based fungicides, avoid overhead watering",
        },
        {
          name: "Angular Leaf Spot",
          risk: "High",
          prevention: "Resistant varieties, crop rotation",
        },
        {
          name: "Bean Rust",
          risk: "Medium",
          prevention: "Fungicide application, good air circulation",
        },
      ],
      lowRainfall: [
        {
          name: "Drought Stress",
          risk: "High",
          prevention: "Mulching, efficient irrigation",
        },
        {
          name: "Spider Mites",
          risk: "Medium",
          prevention: "Maintain humidity, miticide application",
        },
      ],
    },
    recommendations: {
      planting:
        "Plant after last frost. Inoculate seeds with rhizobia for nitrogen fixation. Plant spacing: 30cm x 10cm.",
      care: "Avoid overhead watering to prevent diseases. Stake climbing varieties. Harvest pods regularly for continuous production.",
      harvest:
        "Harvest when pods are full but still tender for fresh consumption. For dry beans, harvest when pods rattle.",
      irrigation:
        "Requires 400-500mm water. Critical periods: flowering and pod filling.",
      fertilizer:
        "Basal: 8:18:8 at 150kg/ha. Minimal nitrogen needed due to nitrogen fixation.",
    },
  },
  soya: {
    name: { en: "Soya Beans", ch: "Soya" },
    emoji: "🌱",
    type: "Protein Crop",
    plantingMonths: [11, 12, 1],
    harvestMonths: [4, 5, 6],
    optimalTemp: { min: 20, max: 30 },
    optimalRainfall: { min: 450, max: 700 },
    diseases: {
      highHumidity: [
        {
          name: "Soybean Rust",
          risk: "Very High",
          prevention: "Triazole fungicides, early detection",
        },
        {
          name: "Frogeye Leaf Spot",
          risk: "High",
          prevention: "Resistant varieties, fungicide rotation",
        },
        {
          name: "Bacterial Blight",
          risk: "Medium",
          prevention: "Pathogen-free seeds, copper sprays",
        },
      ],
      lowRainfall: [
        {
          name: "Drought Stress",
          risk: "High",
          prevention: "Irrigation during critical periods",
        },
        {
          name: "Red Spider Mites",
          risk: "Medium",
          prevention: "Maintain field hygiene, miticides",
        },
      ],
    },
    recommendations: {
      planting:
        "Use certified seeds. Inoculate with appropriate rhizobia strain (Bradyrhizobium japonicum). Plant spacing: 45cm x 5cm.",
      care: "Monitor for rust disease weekly during flowering. Apply foliar fertilizer during flowering for better pod set.",
      harvest:
        "Harvest when pods rattle and leaves turn yellow. Moisture content should be 14-16%.",
      irrigation:
        "Requires 450-700mm water. Critical periods: flowering (R1-R2) and pod filling (R5-R6).",
      fertilizer:
        "Basal: 8:18:8 + 4S at 200kg/ha. Foliar: 0:52:34 during flowering.",
    },
  },
  cassava: {
    name: { en: "Cassava", ch: "Chinangwa" },
    emoji: "🥔",
    type: "Drought Tolerant Crop",
    plantingMonths: [10, 11, 12, 1],
    harvestMonths: [10, 11, 12],
    optimalTemp: { min: 25, max: 35 },
    optimalRainfall: { min: 600, max: 1500 },
    diseases: {
      highHumidity: [
        {
          name: "Cassava Mosaic Disease",
          risk: "High",
          prevention: "Virus-free planting material, vector control",
        },
        {
          name: "Cassava Brown Streak",
          risk: "High",
          prevention: "Clean planting material, whitefly control",
        },
        {
          name: "Bacterial Blight",
          risk: "Medium",
          prevention: "Pathogen-free cuttings, field sanitation",
        },
      ],
      lowRainfall: [
        {
          name: "Drought Tolerance",
          risk: "Low",
          prevention: "Naturally drought tolerant",
        },
        {
          name: "Spider Mites",
          risk: "Low",
          prevention: "Maintain plant vigor",
        },
      ],
    },
    recommendations: {
      planting:
        "Use healthy stem cuttings 20-25cm long. Plant at 45° angle with 2/3 buried. Plant spacing: 1m x 1m.",
      care: "Weed regularly in first 3 months. Minimal fertilizer needed. Stake tall varieties to prevent lodging.",
      harvest:
        "Harvest 12-18 months after planting when leaves start yellowing. Can be left in ground as storage.",
      irrigation:
        "Drought tolerant but benefits from irrigation during establishment.",
      fertilizer:
        "Minimal requirements. Basal: 8:18:8 at 100kg/ha if soil is poor.",
    },
  },
  sweet_potatoes: {
    name: { en: "Sweet Potatoes", ch: "Mbatata" },
    emoji: "🍠",
    type: "Root Crop",
    plantingMonths: [10, 11, 12, 1],
    harvestMonths: [3, 4, 5, 6],
    optimalTemp: { min: 21, max: 26 },
    optimalRainfall: { min: 500, max: 1000 },
    diseases: {
      highHumidity: [
        {
          name: "Sweet Potato Virus Disease",
          risk: "High",
          prevention: "Virus-free planting material",
        },
        {
          name: "Black Rot",
          risk: "Medium",
          prevention: "Proper curing, avoid injuries",
        },
        {
          name: "Soft Rot",
          risk: "Medium",
          prevention: "Good drainage, proper storage",
        },
      ],
      lowRainfall: [
        {
          name: "Drought Stress",
          risk: "Medium",
          prevention: "Mulching, supplemental irrigation",
        },
        {
          name: "Sweet Potato Weevil",
          risk: "High",
          prevention: "Crop rotation, pheromone traps",
        },
      ],
    },
    recommendations: {
      planting:
        "Use disease-free vines 25-30cm long. Plant on ridges for good drainage. Plant spacing: 100cm x 30cm.",
      care: "Hill soil around vines to encourage root development. Control weevils with crop rotation and clean cultivation.",
      harvest:
        "Harvest before first frost. Handle carefully to avoid bruising. Cure in sun for 2-3 days.",
      irrigation: "Requires consistent moisture during root development phase.",
      fertilizer:
        "Basal: 8:18:8 at 200kg/ha. Avoid excess nitrogen which promotes leaf growth over roots.",
    },
  },
  groundnuts: {
    name: { en: "Groundnuts", ch: "Ntedza" },
    emoji: "🥜",
    type: "Legume Crop",
    plantingMonths: [11, 12, 1],
    harvestMonths: [4, 5, 6],
    optimalTemp: { min: 20, max: 30 },
    optimalRainfall: { min: 500, max: 1000 },
    diseases: {
      highHumidity: [
        {
          name: "Early Leaf Spot",
          risk: "High",
          prevention: "Chlorothalonil fungicide, resistant varieties",
        },
        {
          name: "Late Leaf Spot",
          risk: "High",
          prevention: "Regular fungicide applications",
        },
        {
          name: "Rosette Disease",
          risk: "Medium",
          prevention: "Aphid control, resistant varieties",
        },
      ],
      lowRainfall: [
        {
          name: "Drought Stress",
          risk: "High",
          prevention: "Irrigation during pod filling",
        },
        {
          name: "Thrips",
          risk: "Medium",
          prevention: "Insecticide application, field hygiene",
        },
      ],
    },
    recommendations: {
      planting:
        "Plant certified seeds. Apply gypsum (200kg/ha) during flowering for pod development. Plant spacing: 30cm x 10cm.",
      care: "Avoid disturbing plants during pegging stage. Control leaf spot diseases with regular fungicide applications.",
      harvest:
        "Harvest when leaves turn yellow and pods are mature. Dry properly to 8% moisture before storage.",
      irrigation:
        "Requires 500-600mm water. Critical period: pod filling stage.",
      fertilizer:
        "Basal: 8:18:8 at 200kg/ha. Gypsum: 200kg/ha at flowering for calcium.",
    },
  },
};

// Initialize application
document.addEventListener("DOMContentLoaded", () => {
  checkBackendConnection();
  loadCurrentWeather();
  loadWeatherForecast();
  setupEventListeners();
});

// Check backend connection
async function checkBackendConnection() {
  const statusIndicator = document.getElementById("statusIndicator");
  const statusText = document.getElementById("statusText");

  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (response.ok) {
      statusIndicator.className = "status-indicator connected";
      statusText.textContent = "ML Model Connected";
    } else {
      throw new Error("Backend not responding");
    }
  } catch (error) {
    console.error("Backend connection failed:", error);
    statusIndicator.className = "status-indicator error";
    statusText.textContent = "ML Model Offline";
  }
}

// Setup event listeners
function setupEventListeners() {
  const predictionForm = document.getElementById("predictionForm");
  predictionForm.addEventListener("submit", handlePredictionSubmit);
}

// Show page function
function showPage(pageId) {
  const pages = document.querySelectorAll(".page-section");
  pages.forEach((page) => page.classList.remove("active"));

  const targetPage = document.getElementById(`${pageId}Page`);
  if (targetPage) {
    targetPage.classList.add("active");
  }
}

// Show section function
function showSection(sectionId) {
  // Update navigation buttons
  const navButtons = document.querySelectorAll(".nav-button");
  navButtons.forEach((btn) => btn.classList.remove("active"));

  const activeNavButton = document.querySelector(
    `[data-section="${sectionId}"]`
  );
  if (activeNavButton) {
    activeNavButton.classList.add("active");
  }

  // Update content sections
  const contentSections = document.querySelectorAll(".content-section");
  contentSections.forEach((section) => section.classList.remove("active"));

  const targetSection = document.getElementById(`${sectionId}Section`);
  if (targetSection) {
    targetSection.classList.add("active");
  }

  // Load data for specific sections if needed
  if (sectionId === "weatherDetails") {
    loadCurrentWeather();
  } else if (sectionId === "forecast") {
    loadWeatherForecast();
  }
}

// Handle prediction form submission
async function handlePredictionSubmit(e) {
  e.preventDefault();

  const monthSelect = document.getElementById("monthSelect");
  const yearSelect = document.getElementById("yearSelect");
  const cropSelect = document.getElementById("cropSelect");
  const predictButton = document.getElementById("predictButton");

  const month = parseInt(monthSelect.value);
  const year = parseInt(yearSelect.value);
  const crop = cropSelect.value;

  if (!month || !year || !crop) {
    showError(
      "Please select month, year, and crop type for comprehensive analysis."
    );
    return;
  }

  // Store selections globally
  selectedMonth = month;
  selectedCrop = crop;

  // Show loading state
  predictButton.disabled = true;
  predictButton.innerHTML =
    '<i class="fas fa-spinner fa-spin"></i> Generating Professional Analysis...';

  try {
    // Call backend API
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        year: year,
        month: month,
      }),
    });

    const data = await response.json();

    console.log("response: ", data);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const predictions = data;
    currentPredictions = predictions;

    // Update results summary
    updateResultsSummary(crop, month, year);

    // Display predictions
    displayPredictionResults(predictions);

    // Generate farming recommendations
    generateProfessionalFarmingRecommendations(crop, month, predictions);

    // Generate disease alerts
    generateDiseaseAlerts(crop, predictions);

    // Switch to results page
    showPage("results");
  } catch (error) {
    console.error("Prediction failed:", error);
    showError(`Failed to get predictions: ${error.message}`);
  } finally {
    // Reset button
    predictButton.disabled = false;
    predictButton.innerHTML =
      '<i class="fas fa-chart-line"></i> <span>Generate Professional Analysis</span>';
  }
}

// Update results summary
function updateResultsSummary(crop, month, year) {
  const resultsSummary = document.getElementById("resultsSummary");
  const cropData = cropDatabase[crop];
  const monthNames = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];

  resultsSummary.innerHTML = `
          <div class="summary-item">
              <div class="summary-label">Selected Crop</div>
              <div class="summary-value">${cropData.emoji} ${
    cropData.name.en
  }</div>
          </div>
          <div class="summary-item">
              <div class="summary-label">Analysis Month</div>
              <div class="summary-value">${monthNames[month]} ${year}</div>
          </div>
          <div class="summary-item">
              <div class="summary-label">Crop Type</div>
              <div class="summary-value">${cropData.type}</div>
          </div>
          <div class="summary-item">
              <div class="summary-label">Season Status</div>
              <div class="summary-value">${getSeasonStatus(
                cropData,
                month
              )}</div>
          </div>
      `;
}

// Get season status
function getSeasonStatus(cropData, month) {
  if (cropData.plantingMonths.includes(month)) {
    return "🌱 Planting Season";
  } else if (cropData.harvestMonths.includes(month)) {
    return "🌾 Harvest Season";
  } else {
    return "🔧 Maintenance Period";
  }
}

// Display prediction results
function displayPredictionResults(predictions) {
  const predictionsGrid = document.getElementById("predictionsGrid");
  predictionsGrid.innerHTML = "";

  // Map prediction keys to display information
  const predictionMap = {
    tmin: {
      name: "Minimum Temperature",
      unit: "°C",
      icon: "fas fa-thermometer-empty",
      color: "#2563eb",
      getInterpretation: (value) => {
        if (value < 10) return "Very cold conditions may stress crops";
        if (value < 15) return "Cool temperatures, slower crop growth expected";
        if (value > 25) return "Warm nights, favorable for tropical crops";
        return "Moderate temperatures, suitable for most crops";
      },
    },
    tmax: {
      name: "Maximum Temperature",
      unit: "°C",
      icon: "fas fa-thermometer-full",
      color: "#dc2626",
      getInterpretation: (value) => {
        if (value > 35) return "Heat stress risk - increase irrigation";
        if (value > 30) return "Hot conditions - monitor crop water needs";
        if (value < 20) return "Cool conditions may slow growth";
        return "Optimal temperature range for crop development";
      },
    },
    rainfall: {
      name: "Rainfall",
      unit: "mm",
      icon: "fas fa-cloud-rain",
      color: "#3b82f6",
      getInterpretation: (value) => {
        if (value < 20) return "Drought conditions - irrigation required";
        if (value < 50) return "Low rainfall - supplemental watering needed";
        if (value > 200) return "Heavy rainfall - ensure good drainage";
        if (value > 150) return "High rainfall - monitor for waterlogging";
        return "Adequate rainfall for most crop needs";
      },
    },
    humidity: {
      name: "Relative Humidity",
      unit: "%",
      icon: "fas fa-tint",
      color: "#06b6d4",
      getInterpretation: (value) => {
        if (value > 85) return "Very high humidity - disease risk elevated";
        if (value > 75) return "High humidity - monitor for fungal diseases";
        if (value < 40) return "Low humidity - increase irrigation frequency";
        return "Moderate humidity levels";
      },
    },
    wind_speed: {
      name: "Wind Speed",
      unit: "m/s",
      icon: "fas fa-wind",
      color: "#6b7280",
      getInterpretation: (value) => {
        if (value > 15) return "Strong winds - provide crop support";
        if (value > 10) return "Moderate winds - good for air circulation";
        if (value < 2) return "Low wind - may increase disease pressure";
        return "Normal wind conditions";
      },
    },
  };

  Object.entries(predictions).forEach(([key, value]) => {
    const info = predictionMap[key];
    if (!info) return;

    const card = document.createElement("div");
    card.className = "prediction-card";

    card.innerHTML = `
              <div class="card-header">
                  <div class="card-title">${info.name}</div>
                  <div class="card-icon"><i class="${
                    info.icon
                  }" style="color: ${info.color}"></i></div>
              </div>
              <div class="prediction-value" style="color: ${
                info.color
              }">${Number(value).toFixed(1)}</div>
              <div class="prediction-unit">${info.unit}</div>
              <div class="prediction-interpretation">
                  ${info.getInterpretation(value)}
              </div>
          `;

    predictionsGrid.appendChild(card);
  });
}

// Generate professional farming recommendations
function generateProfessionalFarmingRecommendations(crop, month, predictions) {
  const recommendationsContent = document.getElementById(
    "recommendationsContent"
  );

  if (!cropDatabase[crop]) {
    recommendationsContent.innerHTML = `
              <div class="error-message">
                  <i class="fas fa-exclamation-triangle"></i>
                  Crop information not available for ${crop}
              </div>
          `;
    return;
  }

  const cropData = cropDatabase[crop];
  const monthNames = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];

  let html = "";

  // 1. Current Month Analysis
  html += generateCurrentMonthAnalysis(
    cropData,
    month,
    predictions,
    monthNames[month]
  );

  // 2. Weather-Based Recommendations
  html += generateWeatherBasedRecommendations(cropData, predictions);

  // 3. Crop Management Practices
  html += generateCropManagementPractices(cropData, month);

  // 4. Market and Post-Harvest Advice
  html += generateMarketAdvice(cropData, month);

  recommendationsContent.innerHTML = html;
}

// Generate current month analysis
function generateCurrentMonthAnalysis(cropData, month, predictions, monthName) {
  let priority = "medium";
  let analysis = "";
  let recommendations = [];

  if (cropData.plantingMonths.includes(month)) {
    priority = "high";
    analysis = `${monthName} is an optimal planting month for ${cropData.name.en}. Current weather predictions show favorable conditions for seed germination and early growth.`;
    recommendations = [
      "Prepare land and acquire certified seeds",
      "Apply basal fertilizer as recommended",
      "Ensure proper seed spacing and depth",
      "Monitor soil moisture for optimal germination",
    ];
  } else if (cropData.harvestMonths.includes(month)) {
    priority = "high";
    analysis = `${monthName} is harvest season for ${cropData.name.en}. Focus on proper harvesting techniques and post-harvest handling to maximize quality and reduce losses.`;
    recommendations = [
      "Monitor crop maturity indicators",
      "Prepare harvesting equipment and storage",
      "Plan for proper drying and curing",
      "Arrange transportation and marketing",
    ];
  } else {
    priority = "low";
    analysis = `${monthName} is not a primary planting or harvest month for ${cropData.name.en}. Focus on field preparation, maintenance, or planning for the next season.`;
    recommendations = [
      "Conduct soil testing and improvement",
      "Plan crop rotation strategies",
      "Maintain farm equipment",
      "Attend agricultural training programs",
    ];
  }

  return `
          <div class="recommendation-category">
              <div class="category-header">
                  <i class="fas fa-calendar-check category-icon"></i>
                  <h3 class="category-title">${monthName} Analysis for ${
    cropData.name.en
  }</h3>
              </div>
              <div class="recommendation-card">
                  <div class="recommendation-header">
                      <div class="recommendation-title">
                          <i class="fas fa-chart-line"></i>
                          Current Month Assessment
                      </div>
                      <div class="priority-badge priority-${priority}">
                          ${priority.toUpperCase()} PRIORITY
                      </div>
                  </div>
                  <div class="recommendation-content">
                      <p>${analysis}</p>
                      <div class="recommendation-details">
                          <h5><i class="fas fa-tasks"></i> Key Actions for ${monthName}:</h5>
                          <ul class="recommendation-list">
                              ${recommendations
                                .map((rec) => `<li>${rec}</li>`)
                                .join("")}
                          </ul>
                      </div>
                  </div>
              </div>
          </div>
      `;
}

// Generate weather-based recommendations
function generateWeatherBasedRecommendations(cropData, predictions) {
  const recommendations = [];

  // Temperature recommendations
  if (predictions.tmax > cropData.optimalTemp.max) {
    recommendations.push({
      title: "High Temperature Management",
      priority: "high",
      icon: "fas fa-thermometer-full",
      content: `Predicted maximum temperature (${predictions.tmax.toFixed(
        1
      )}°C) exceeds optimal range for ${cropData.name.en}.`,
      actions: [
        "Increase irrigation frequency during hot periods",
        "Apply mulch to reduce soil temperature",
        "Consider shade nets for sensitive crops",
        "Schedule field work for early morning or evening",
        "Monitor plants for heat stress symptoms",
      ],
    });
  }

  if (predictions.tmin < cropData.optimalTemp.min) {
    recommendations.push({
      title: "Cold Protection Measures",
      priority: "medium",
      icon: "fas fa-thermometer-empty",
      content: `Predicted minimum temperature (${predictions.tmin.toFixed(
        1
      )}°C) is below optimal range.`,
      actions: [
        "Use row covers or plastic tunnels for protection",
        "Delay planting if temperatures are too low",
        "Ensure good drainage to prevent frost damage",
        "Consider cold-tolerant varieties",
      ],
    });
  }

  // Rainfall recommendations
  if (predictions.rainfall < cropData.optimalRainfall.min) {
    recommendations.push({
      title: "Irrigation Planning Required",
      priority: "high",
      icon: "fas fa-tint",
      content: `Predicted rainfall (${predictions.rainfall.toFixed(
        1
      )}mm) is below crop requirements.`,
      actions: [
        "Install drip irrigation systems if possible",
        "Practice water conservation techniques",
        "Apply organic mulch to retain soil moisture",
        "Consider drought-tolerant varieties",
        "Schedule irrigation during critical growth stages",
      ],
    });
  }

  if (predictions.rainfall > cropData.optimalRainfall.max) {
    recommendations.push({
      title: "Drainage Management Critical",
      priority: "high",
      icon: "fas fa-cloud-rain",
      content: `Predicted rainfall (${predictions.rainfall.toFixed(
        1
      )}mm) may cause waterlogging.`,
      actions: [
        "Improve field drainage systems",
        "Create raised beds for better drainage",
        "Monitor for root rot diseases",
        "Avoid field operations when soil is waterlogged",
        "Apply fungicides preventively if needed",
      ],
    });
  }

  // Humidity recommendations
  if (predictions.humidity > 80) {
    recommendations.push({
      title: "High Humidity Disease Prevention",
      priority: "high",
      icon: "fas fa-eye-dropper",
      content: `High humidity (${predictions.humidity.toFixed(
        1
      )}%) increases disease risk.`,
      actions: [
        "Ensure good air circulation in crop canopy",
        "Apply preventive fungicides",
        "Avoid overhead irrigation",
        "Remove infected plant material immediately",
        "Increase plant spacing if possible",
      ],
    });
  }

  let html = `
          <div class="recommendation-category">
              <div class="category-header">
                  <i class="fas fa-cloud-sun category-icon"></i>
                  <h3 class="category-title">Weather-Based Management Recommendations</h3>
              </div>
      `;

  recommendations.forEach((rec) => {
    html += `
              <div class="recommendation-card">
                  <div class="recommendation-header">
                      <div class="recommendation-title">
                          <i class="${rec.icon}"></i>
                          ${rec.title}
                      </div>
                      <div class="priority-badge priority-${rec.priority}">
                          ${rec.priority.toUpperCase()}
                      </div>
                  </div>
                  <div class="recommendation-content">
                      <p>${rec.content}</p>
                      <div class="recommendation-details">
                          <h5><i class="fas fa-cogs"></i> Recommended Actions:</h5>
                          <ul class="recommendation-list">
                              ${rec.actions
                                .map((action) => `<li>${action}</li>`)
                                .join("")}
                          </ul>
                      </div>
                  </div>
              </div>
          `;
  });

  html += "</div>";
  return html;
}

// Generate crop management practices
function generateCropManagementPractices(cropData, month) {
  return `
          <div class="recommendation-category">
              <div class="category-header">
                  <i class="fas fa-tools category-icon"></i>
                  <h3 class="category-title">Professional Crop Management Practices</h3>
              </div>
              <div class="recommendation-card">
                  <div class="recommendation-header">
                      <div class="recommendation-title">
                          <i class="fas fa-seedling"></i>
                          Planting Guidelines
                      </div>
                     
                  </div>
                  <div class="recommendation-content">
                      <p>${cropData.recommendations.planting}</p>
                  </div>
              </div>
              <div class="recommendation-card">
                  <div class="recommendation-header">
                      <div class="recommendation-title">
                          <i class="fas fa-leaf"></i>
                          Crop Care & Maintenance
                      </div>
                     
                  </div>
                  <div class="recommendation-content">
                      <p>${cropData.recommendations.care}</p>
                  </div>
              </div>
              <div class="recommendation-card">
                  <div class="recommendation-header">
                      <div class="recommendation-title">
                          <i class="fas fa-flask"></i>
                          Fertilizer Application
                      </div>
                     
                  </div>
                  <div class="recommendation-content">
                      <p>${cropData.recommendations.fertilizer}</p>
                  </div>
              </div>
              <div class="recommendation-card">
                  <div class="recommendation-header">
                      <div class="recommendation-title">
                          <i class="fas fa-tint"></i>
                          Irrigation Management
                      </div>
                     
                  </div>
                  <div class="recommendation-content">
                      <p>${cropData.recommendations.irrigation}</p>
                  </div>
              </div>
          </div>
      `;
}

// Generate market advice
function generateMarketAdvice(cropData, month) {
  let marketAdvice = "";
  let harvestAdvice = cropData.recommendations.harvest;

  if (cropData.harvestMonths.includes(month)) {
    marketAdvice = `This is harvest season for ${cropData.name.en}. Monitor market prices and consider storage options for better prices. Ensure proper post-harvest handling to maintain quality and reduce losses.`;
  } else if (cropData.plantingMonths.includes(month)) {
    marketAdvice = `Plan your production based on market demand. Consider contract farming opportunities and ensure you have access to quality inputs and technical support.`;
  } else {
    marketAdvice = `Use this time to research market trends, build relationships with buyers, and plan your marketing strategy for the upcoming season.`;
  }

  return `
          <div class="recommendation-category">
              <div class="category-header">
                  <i class="fas fa-chart-line category-icon"></i>
                  <h3 class="category-title">Market Intelligence & Post-Harvest</h3>
              </div>
              <div class="recommendation-card">
                  <div class="recommendation-header">
                      <div class="recommendation-title">
                          <i class="fas fa-harvest"></i>
                          Harvest Guidelines
                      </div>
                    
                  </div>
                  <div class="recommendation-content">
                      <p>${harvestAdvice}</p>
                  </div>
              </div>
              <div class="recommendation-card">
                  <div class="recommendation-header">
                      <div class="recommendation-title">
                          <i class="fas fa-coins"></i>
                          Market Strategy
                      </div>
                     
                  </div>
                  <div class="recommendation-content">
                      <p>${marketAdvice}</p>
                      <div class="recommendation-details">
                          <h5><i class="fas fa-lightbulb"></i> Market Tips:</h5>
                          <ul class="recommendation-list">
                              <li>Join farmer cooperatives for better bargaining power</li>
                              <li>Maintain quality standards for premium prices</li>
                              <li>Consider value addition opportunities</li>
                              <li>Keep records of production costs and revenues</li>
                              <li>Diversify crops to spread market risks</li>
                          </ul>
                      </div>
                  </div>
              </div>
          </div>
      `;
}

// Generate disease alerts
function generateDiseaseAlerts(crop, predictions) {
  const diseaseRiskLevel = document.getElementById("diseaseRiskLevel");
  const diseaseList = document.getElementById("diseaseList");

  if (!cropDatabase[crop] || !predictions) {
    diseaseRiskLevel.className = "disease-risk-level disease-risk-low";
    diseaseRiskLevel.innerHTML = `
              <i class="fas fa-shield-alt"></i>
              <span>Disease Risk Assessment: Data Not Available</span>
          `;
    diseaseList.innerHTML = `
              <div class="disease-item">
                  <div class="disease-name">Assessment Unavailable</div>
                  <div class="disease-risk">Status: Incomplete Data</div>
                  <div class="disease-prevention">
                      <strong>Note:</strong> Complete the prediction analysis to receive detailed disease risk assessment.
                  </div>
              </div>
          `;
    return;
  }

  const cropData = cropDatabase[crop];
  let riskLevel = "Low";
  let riskClass = "disease-risk-low";
  let diseases = [];

  // Determine disease risk based on weather conditions
  if (predictions.humidity > 75 && predictions.rainfall > 50) {
    riskLevel = "High";
    riskClass = "disease-risk-high";
    diseases = cropData.diseases.highHumidity;
  } else if (predictions.rainfall < 30) {
    riskLevel = "Medium";
    riskClass = "disease-risk-medium";
    diseases = cropData.diseases.lowRainfall;
  } else if (predictions.rainfall > 150) {
    riskLevel = "High";
    riskClass = "disease-risk-high";
    diseases = cropData.diseases.highRainfall || cropData.diseases.highHumidity;
  } else {
    riskLevel = "Low";
    riskClass = "disease-risk-low";
    diseases = [
      {
        name: "General monitoring recommended",
        risk: "Low",
        prevention:
          "Continue regular field inspections and maintain good agricultural practices. Weather conditions are favorable with low disease pressure expected.",
      },
    ];
  }

  // Update risk level display
  diseaseRiskLevel.className = `disease-risk-level ${riskClass}`;
  diseaseRiskLevel.innerHTML = `
          <i class="fas fa-shield-alt"></i>
          <span>Disease Risk Level: ${riskLevel} for ${cropData.name.en}</span>
      `;

  // Update disease list
  diseaseList.innerHTML = diseases
    .map(
      (disease) => `
          <div class="disease-item">
              <div class="disease-name">${disease.name}</div>
              <div class="disease-risk">Risk Level: ${disease.risk}</div>
              <div class="disease-prevention">
                  <strong>Prevention Strategy:</strong> ${disease.prevention}
              </div>
          </div>
      `
    )
    .join("");
}

// Load current weather from OpenWeather API
async function loadCurrentWeather() {
  const weatherDetails = document.getElementById("weatherDetails");

  console.log("Loading current weather...");

  try {
    // Show loading state
    weatherDetails.innerHTML = `
              <div class="loading">
                  <div class="loading-spinner"></div>
                  <p>Loading current weather data...</p>
              </div>
          `;

    const response = await fetch(
      `https://api.openweathermap.org/data/2.5/weather?lat=${KASUNGU_COORDS.lat}&lon=${KASUNGU_COORDS.lon}&units=metric&appid=${OPENWEATHER_API_KEY}`
    );

    //console.log("Current Weather: ", await response.json());

    if (!response.ok) {
      throw new Error("Weather API request failed");
    }

    const data = await response.json();
    currentWeatherData = data;
    console.log(currentWeatherData);
    updateCurrentWeather(data);
  } catch (error) {
    console.error("Error fetching current weather:", error);
    // Show fallback data
    showFallbackCurrentWeather();
  }
}

// Show fallback current weather data
function showFallbackCurrentWeather() {
  document.getElementById("weatherLocation").textContent =
    "Kasungu District, MW";
  document.getElementById("weatherTemp").textContent = "28°C";
  document.getElementById("weatherDesc").textContent = "Partly Cloudy";
  document.getElementById("weatherIcon").innerHTML =
    '<i class="fas fa-cloud-sun" style="font-size: 5rem;"></i>';

  const weatherDetails = document.getElementById("weatherDetails");
  weatherDetails.innerHTML = `
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-thermometer-half"></i></div>
              <div class="weather-detail-value">30°C</div>
              <div class="weather-detail-label">Feels Like</div>
          </div>
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-tint"></i></div>
              <div class="weather-detail-value">65%</div>
              <div class="weather-detail-label">Humidity</div>
          </div>
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-wind"></i></div>
              <div class="weather-detail-value">12 km/h</div>
              <div class="weather-detail-label">Wind Speed</div>
          </div>
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-eye"></i></div>
              <div class="weather-detail-value">10 km</div>
              <div class="weather-detail-label">Visibility</div>
          </div>
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-compress-arrows-alt"></i></div>
              <div class="weather-detail-value">1013 hPa</div>
              <div class="weather-detail-label">Pressure</div>
          </div>
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-sun"></i></div>
              <div class="weather-detail-value">06:30</div>
              <div class="weather-detail-label">Sunrise</div>
          </div>
      `;
}

// Load 5-day weather forecast
async function loadWeatherForecast() {
  const forecastList = document.getElementById("forecastList");

  try {
    // Show loading state
    forecastList.innerHTML = `
              <div class="loading">
                  <div class="loading-spinner"></div>
                  <p>Loading weather forecast...</p>
              </div>
          `;

    const response = await fetch(
      `https://api.openweathermap.org/data/2.5/forecast?lat=${KASUNGU_COORDS.lat}&lon=${KASUNGU_COORDS.lon}&units=metric&appid=${OPENWEATHER_API_KEY}`
    );

    if (!response.ok) {
      throw new Error("Forecast API request failed");
    }

    const data = await response.json();
    forecastData = data;
    updateWeatherForecast(data);
  } catch (error) {
    console.error("Error fetching weather forecast:", error);
    // Show fallback forecast
    showFallbackForecast();
  }
}

// Show fallback forecast data
function showFallbackForecast() {
  const forecastList = document.getElementById("forecastList");
  const fallbackForecast = [
    {
      day: "Today",
      date: "Jan 25",
      high: 32,
      low: 22,
      condition: "Partly Cloudy",
      icon: "fas fa-cloud-sun",
    },
    {
      day: "Tomorrow",
      date: "Jan 26",
      high: 30,
      low: 20,
      condition: "Sunny",
      icon: "fas fa-sun",
    },
    {
      day: "Sunday",
      date: "Jan 27",
      high: 28,
      low: 18,
      condition: "Light Rain",
      icon: "fas fa-cloud-rain",
    },
    {
      day: "Monday",
      date: "Jan 28",
      high: 29,
      low: 19,
      condition: "Cloudy",
      icon: "fas fa-cloud",
    },
    {
      day: "Tuesday",
      date: "Jan 29",
      high: 31,
      low: 21,
      condition: "Partly Cloudy",
      icon: "fas fa-cloud-sun",
    },
  ];

  forecastList.innerHTML = fallbackForecast
    .map(
      (day) => `
          <div class="forecast-item">
              <div class="forecast-day">
                  <strong>${day.day}</strong><br>
                  <small>${day.date}</small>
              </div>
              <div class="forecast-weather">
                  <i class="${day.icon}" style="font-size: 2.5rem; color: var(--primary);"></i>
                  <div style="text-align: center;">
                      <div>${day.condition}</div>
                      <small style="color: var(--gray-500);">
                          Typical conditions for Kasungu
                      </small>
                  </div>
              </div>
              <div class="forecast-temps">
                  <span class="temp-high">${day.high}°</span>
                  <span class="temp-low">${day.low}°</span>
              </div>
          </div>
      `
    )
    .join("");
}

// Update current weather display
/*
function updateCurrentWeather(data) {
  const location = `${data.name}, ${data.sys.country}`;
  const temp = `${Math.round(data.main.temp)}°C`;
  const description = data.weather[0].description;
  const iconCode = data.weather[0].icon;

  document.getElementById("weatherLocation").textContent = location;
  document.getElementById("weatherTemp").textContent = temp;
  document.getElementById("weatherDesc").textContent =
    description.charAt(0).toUpperCase() + description.slice(1);

  const iconUrl = `https://openweathermap.org/img/wn/${iconCode}@2x.png`;
  document.getElementById(
    "weatherIcon"
  ).innerHTML = `<img src="${iconUrl}" alt="${description}" style="width: 80px; height: 80px;" />`;

  // Update weather details
  const weatherDetails = document.getElementById("weatherDetails");
  weatherDetails.innerHTML = `
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-thermometer-half"></i></div>
              <div class="weather-detail-value">${Math.round(
                data.main.feels_like
              )}°C</div>
              <div class="weather-detail-label">Feels Like</div>
          </div>
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-tint"></i></div>
              <div class="weather-detail-value">${
                data.main.humidity
              }%</div>
              <div class="weather-detail-label">Humidity</div>
          </div>
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-wind"></i></div>
              <div class="weather-detail-value">${Math.round(
                data.wind.speed * 3.6
              )} km/h</div>
              <div class="weather-detail-label">Wind Speed</div>
          </div>
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-eye"></i></div>
              <div class="weather-detail-value">${(
                data.visibility / 1000
              ).toFixed(1)} km</div>
              <div class="weather-detail-label">Visibility</div>
          </div>
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-compress-arrows-alt"></i></div>
              <div class="weather-detail-value">${
                data.main.pressure
              } hPa</div>
              <div class="weather-detail-label">Pressure</div>
          </div>
          <div class="weather-detail">
              <div class="weather-detail-icon"><i class="fas fa-sun"></i></div>
              <div class="weather-detail-value">${new Date(
                data.sys.sunrise * 1000
              ).toLocaleTimeString("en-US", {
                hour: "2-digit",
                minute: "2-digit",
              })}</div>
              <div class="weather-detail-label">Sunrise</div>
          </div>
      `;
}
*/

function updateCurrentWeather(data) {
  console.log("Updating current weather... ");
  const location = `${data.name}, ${data.sys.country}`;
  const temp = `${Math.round(data.main.temp)}°C`;
  const description = data.weather[0].description;
  const iconCode = data.weather[0].icon;

  // Update location, temperature, and description
  document.getElementById("weatherLocation").textContent = location;
  document.getElementById("weatherTemp").textContent = temp;
  document.getElementById("weatherDesc").textContent =
    description.charAt(0).toUpperCase() + description.slice(1);

  // Set weather icon
  const iconUrl = `https://openweathermap.org/img/wn/${iconCode}@2x.png`;
  document.getElementById("weatherIcon").innerHTML = `
<img src="${iconUrl}" alt="${description}" style="width: 80px; height: 80px;" />
`;

  // Update weather details
  const weatherDetails = document.getElementById("weatherDetails");
  weatherDetails.innerHTML = `
<div class="weather-detail">
<div class="weather-detail-icon"><i class="fas fa-thermometer-half"></i></div>
<div class="weather-detail-value">${Math.round(data.main.feels_like)}°C</div>
<div class="weather-detail-label">Feels Like</div>
</div>
<div class="weather-detail">
<div class="weather-detail-icon"><i class="fas fa-tint"></i></div>
<div class="weather-detail-value">${data.main.humidity}%</div>
<div class="weather-detail-label">Humidity</div>
</div>
<div class="weather-detail">
<div class="weather-detail-icon"><i class="fas fa-wind"></i></div>
<div class="weather-detail-value">${Math.round(
    data.wind.speed * 3.6
  )} km/h</div>
<div class="weather-detail-label">Wind Speed</div>
</div>
<div class="weather-detail">
<div class="weather-detail-icon"><i class="fas fa-eye"></i></div>
<div class="weather-detail-value">${(data.visibility / 1000).toFixed(
    1
  )} km</div>
<div class="weather-detail-label">Visibility</div>
</div>
<div class="weather-detail">
<div class="weather-detail-icon"><i class="fas fa-compress-arrows-alt"></i></div>
<div class="weather-detail-value">${data.main.pressure} hPa</div>
<div class="weather-detail-label">Pressure</div>
</div>
<div class="weather-detail">
<div class="weather-detail-icon"><i class="fas fa-water"></i></div>
<div class="weather-detail-value">${data.main.sea_level || "—"} hPa</div>
<div class="weather-detail-label">Sea Level</div>
</div>
<div class="weather-detail">
<div class="weather-detail-icon"><i class="fas fa-mountain"></i></div>
<div class="weather-detail-value">${data.main.grnd_level || "—"} hPa</div>
<div class="weather-detail-label">Ground Level</div>
</div>
<div class="weather-detail">
<div class="weather-detail-icon"><i class="fas fa-sun"></i></div>
<div class="weather-detail-value">${new Date(
    data.sys.sunrise * 1000
  ).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  })}</div>
<div class="weather-detail-label">Sunrise</div>
</div>
<div class="weather-detail">
<div class="weather-detail-icon"><i class="fas fa-moon"></i></div>
<div class="weather-detail-value">${new Date(
    data.sys.sunset * 1000
  ).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  })}</div>
<div class="weather-detail-label">Sunset</div>
</div>
`;
}

// Update 5-day weather forecast
function updateWeatherForecast(data) {
  const forecastList = document.getElementById("forecastList");
  const dailyForecasts = {};

  // Group forecasts by day
  data.list.forEach((item) => {
    const date = new Date(item.dt * 1000);
    const dayKey = date.toDateString();

    if (!dailyForecasts[dayKey]) {
      dailyForecasts[dayKey] = {
        date: date,
        temps: [],
        weather: [],
        icons: [],
        humidity: [],
        wind: [],
      };
    }

    dailyForecasts[dayKey].temps.push(item.main.temp);
    dailyForecasts[dayKey].weather.push(item.weather[0].description);
    dailyForecasts[dayKey].icons.push(item.weather[0].icon);
    dailyForecasts[dayKey].humidity.push(item.main.humidity);
    dailyForecasts[dayKey].wind.push(item.wind.speed);
  });

  // Create forecast items for next 5 days
  const days = Object.values(dailyForecasts).slice(0, 5);

  forecastList.innerHTML = days
    .map((day) => {
      const minTemp = Math.min(...day.temps);
      const maxTemp = Math.max(...day.temps);
      const avgHumidity = Math.round(
        day.humidity.reduce((a, b) => a + b, 0) / day.humidity.length
      );
      const avgWind = Math.round(
        (day.wind.reduce((a, b) => a + b, 0) / day.wind.length) * 3.6
      ); // Convert to km/h
      const mostCommonIcon = getMostFrequent(day.icons);
      const mostCommonWeather = getMostFrequent(day.weather);

      const dayName = day.date.toLocaleDateString("en-US", {
        weekday: "long",
      });
      const dateStr = day.date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      });

      return `
              <div class="forecast-item">
                  <div class="forecast-day">
                      <strong>${dayName}</strong><br>
                      <small>${dateStr}</small>
                  </div>
                  <div class="forecast-weather">
                      <img src="https://openweathermap.org/img/wn/${mostCommonIcon}@2x.png" 
                           alt="${mostCommonWeather}" class="forecast-icon">
                      <div style="text-align: center;">
                          <div>${
                            mostCommonWeather.charAt(0).toUpperCase() +
                            mostCommonWeather.slice(1)
                          }</div>
                          <small style="color: var(--gray-500);">
                              ${avgHumidity}% humidity, ${avgWind} km/h wind
                          </small>
                      </div>
                  </div>
                  <div class="forecast-temps">
                      <span class="temp-high">${Math.round(maxTemp)}°</span>
                      <span class="temp-low">${Math.round(minTemp)}°</span>
                  </div>
              </div>
          `;
    })
    .join("");
}

// Utility function to get most frequent item in array
function getMostFrequent(arr) {
  return arr.sort(
    (a, b) =>
      arr.filter((v) => v === b).length - arr.filter((v) => v === a).length
  )[0];
}

// Show error message
function showError(message) {
  const predictionsGrid = document.getElementById("predictionsGrid");
  predictionsGrid.innerHTML = `
          <div class="error-message">
              <i class="fas fa-exclamation-triangle"></i>
              ${message}
          </div>
      `;
}
