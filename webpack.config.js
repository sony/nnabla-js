var path = require('path');

module.exports = {
  mode: "development",
  entry: "./src/index.ts",
  output: {
    filename: 'index.js',
    path: path.resolve(__dirname, 'dist'),
    library: "NNP",
    libraryExport: "default",
    libraryTarget: "umd",
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
      }
    ]
  },
  resolve: {
    extensions: [
      '.ts', '.js',
    ],
  },
};
