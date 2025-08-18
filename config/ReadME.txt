{
  "surgery": "[name of the physical parameters set-up]",
  "contributing_pixels": ["pixel1", "pixel2", "pixel3", ..], # input for transmitter.py -- meaningful data to be loaded
  "stiffnessArray": ["secondK", "thirdK"], # input for transmitter.py -- stiffness array for each "contributing_pixel"

  "model": "2D", # dimensionality of the topology of the sound model: 1D, 2D or 3D
  "geometry": {
    "1D": {
      "numNodes": (int) number of masses along the Y direction,
      "dist": (float) distance between 2 masses,
      "massesRadius": (float) radius of the masses,
      "firstLayerNumNodes": 3,
      "secondLayerNumNodes": 13,
      "thirdLayerNumNodes": 11,
      "drivers": ["m_1", "m_10", "m_20"],
      "listeners": ["m_3", "m_13", "m_23"]},
    "2D": {
      "numNodes": 26,
      "distance": 10,
      "massesRadius": 3.0,
      "dx": 3,
      "dz": 1,
      "firstLayerNumNodes": 10,
      "secondLayerNumNodes": 3,
      "thirdLayerNumNodes": 13,
      "drivers": ["m_1_1_0", "m_1_13_0", "m_1_10_0", "m_1_20_0", "m_1_21_0", "m_1_24_0"],
      "listeners": ["m_1_3_0", "m_1_14_0", "m_1_23_0"]},
    "3D": {
      "numNodes": 26,
      "distance": 10,
      "massesRadius": 3.0,
      "dx": 3,
      "dz": 3,
      "firstLayerNumNodes": 10,
      "secondLayerNumNodes": 3,
      "thirdLayerNumNodes": 13,
      "drivers": ["m_1_1_1", "m_1_13_1", "m_1_10_1", "m_1_20_1", "m_1_21_1", "m_1_24_1"],
      "listeners": ["m_1_3_1", "m_1_14_1", "m_1_23_1"]}
  },
  "parameters": {
    "1D": {
      "acoustic_scaling_factor": 2,
      "US-first-trial": {
        "firstK": 0.5,
        "secondK": 4.4,
        "thirdK": 15.0,
        "firstM": 50.0,
        "secondM": 96.15,
        "thirdM": 96.15
      }
    },
    "2D": {
      "acoustic_scaling_factor": 2,
      [surgery]: {
        "firstK": 0.5,
        "secondK": 4.4,
        "thirdK": 15.0,
        "firstM": 50.0,
        "secondM": 96.15,
        "thirdM": 96.15
      }
    },
    "3D": {
      "acoustic_scaling_factor": 2,
      [surgery]: {
        "firstK": 0.5,
        "secondK": 4.4,
        "thirdK": 15.0,
        "firstM": 50.0,
        "secondM": 96.15,
        "thirdM": 96.15
      }
    }
  }
}
