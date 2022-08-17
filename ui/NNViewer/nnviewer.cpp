#include "nnviewer.h"

#include <QAction>
#include <QDebug>
#include <QFileDialog>
#include <QIcon>
#include <QMenu>
#include <QTableWidgetItem>

#include "./ui_nnviewer.h"

NNViewer::NNViewer(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::NNViewer) {
  ui->setupUi(this);
  ui->treeNN->setHeaderLabels(QStringList() << "Neuron"
                                            << "State");
  ui->tableWeights->horizontalHeader()->setStretchLastSection(true);

  populateMainMenu();
}

NNViewer::~NNViewer() {
  delete ui;
}

void NNViewer::populateMainMenu() {
  QMenu* fileMenu = ui->menubar->addMenu("File");
  QAction* openNNConfigAction = fileMenu->addAction("Open NN config file...");
  QAction* saveNNConfigAction = fileMenu->addAction("Save NN config file...");
  connect(openNNConfigAction, &QAction::triggered, this,
          &NNViewer::openNNConfigFile);

  connect(saveNNConfigAction, &QAction::triggered, this,
          &NNViewer::saveNNConfigFile);
}

cmNN::NeuralNetworkConfiguration NNViewer::deleteNeuron(size_t layer,
                                                        size_t neuron) {
  cmNN::NeuralNetworkConfiguration newConfig;

  newConfig.nLayers = config.nLayers;
  newConfig.maxW = config.maxW;
  newConfig.minW = config.minW;
  newConfig.nInputs = config.nInputs;
  newConfig.neuronsPerLayer = new size_t[config.nLayers];

  for (size_t l = 0; l < config.nLayers; l += 1) {
    size_t tn = config.neuronsPerLayer[l];
    tn = l == layer ? tn - 1 : tn;
    newConfig.neuronsPerLayer[l] = tn;
  }

  size_t numW = cmNN::NeuralNetwork::calculateNumberOfWeights(&newConfig);
  newConfig.nWeights = numW;
  newConfig.weights = new double[numW];

  size_t w = 0;
  for (size_t l = 0; l < config.nLayers; l += 1) {
    for (size_t n = 0; n < config.neuronsPerLayer[l]; n += 1) {
      if (l == layer && n == neuron)
        continue;
      size_t wfn = getNumOfWeightsForNeuron(l, n);
      size_t index = getIndexFirstWeightForNeuron(l, n);
      double* oldW = &config.weights[index];
      for (size_t cw = 0; cw < wfn; cw += 1) {
        if (l == layer + 1 && cw == neuron)
          continue;
        newConfig.weights[w] = oldW[cw];
        w += 1;
      }
    }
  }

  return newConfig;
}

size_t NNViewer::getNumOfWeightsForNeuron(size_t layer, size_t neuron) {
  size_t nWeigths =
      layer == 0 ? config.nInputs : config.neuronsPerLayer[layer - 1];
  nWeigths += 3;

  return nWeigths;
}

size_t NNViewer::getIndexFirstWeightForNeuron(size_t layer, size_t neuron) {
  size_t nWeigths = getNumOfWeightsForNeuron(layer, neuron);

  size_t offset = 0;
  for (size_t l = 0; l < layer; l += 1) {
    size_t prev = getNumOfWeightsForNeuron(l, 0);
    offset += config.neuronsPerLayer[l] * prev;
  }

  offset += (neuron * nWeigths);

  qDebug() << "Offset: " << QString::number(nWeigths);

  return offset;
}

bool NNViewer::isNeuronActive(size_t layer, size_t neuron) {
  size_t nw = getNumOfWeightsForNeuron(layer, neuron);
  size_t fw = getIndexFirstWeightForNeuron(layer, neuron);

  double* w = &config.weights[fw];
  return w[nw - 2] > 0.0;
}

void NNViewer::setIsNeuronActive(size_t layer, size_t neuron, bool isActive) {
  size_t nw = getNumOfWeightsForNeuron(layer, neuron);
  size_t fw = getIndexFirstWeightForNeuron(layer, neuron);

  double* w = &config.weights[fw];
  w[nw - 2] = isActive ? 1.0 : 0.0;
}

void NNViewer::setNeuronWeight(size_t layer,
                               size_t neuron,
                               size_t wIndex,
                               QString newWeight) {
  size_t fw = getIndexFirstWeightForNeuron(layer, neuron);

  double* w = &config.weights[fw];
  qDebug() << "New w: " << newWeight;
  w[wIndex] = newWeight.toDouble();
}

void NNViewer::populateNNTree() {
  ui->treeNN->clear();
  for (size_t l = 0; l < config.nLayers; l += 1) {
    QString name = QString::number(l) + ": ";

    if (l == 0)
      name += "Input";
    else if (l == config.nLayers - 1)
      name += "Output";
    else
      name += "Hidden";

    nnTreeAddRootNode(l, name,
                      QString::number(config.neuronsPerLayer[l]) + " neurons");
  }
}

void NNViewer::populateCmbActivationFx() {
  ui->cmbActFx->clear();
  cmNN::NN_ACTIVATION_FX c = cmNN::NN_ACTIVATION_FX::COUNT;
  for (int f = 0; f < static_cast<int>(c); f += 1) {
    ui->cmbActFx->addItem(cmNN::getActivationFunctionName(
        static_cast<cmNN::NN_ACTIVATION_FX>(f)));
  }
}

void NNViewer::populateLayerDetails(size_t layer) {
  if (layer == cLayer)
    return;

  cLayer = layer;
  qDebug() << "Populating layer: " << QString::number(layer);
}

void NNViewer::populateNeuronDetails(size_t layer, size_t neuron) {
  loadingW = true;
  cNeuron = neuron;
  ui->tableWeights->setRowCount(0);
  size_t nWeigths = getNumOfWeightsForNeuron(layer, neuron);
  size_t firstWeigth = getIndexFirstWeightForNeuron(layer, neuron);

  double* weights = &config.weights[firstWeigth];
  bool isActive = weights[nWeigths - 2] > 0;

  ui->chkIsActive->setChecked(isActive);
  int fx = cmNN::getActivationFunctionIndex(weights[nWeigths - 1], config.minW,
                                            config.maxW);
  ui->txtBias->setText(QString::number(weights[nWeigths - 3], 'f', 15));
  ui->cmbActFx->setCurrentIndex(fx);
  for (size_t w = 0; w < nWeigths - 3; w += 1) {
    QTableWidgetItem* row =
        new QTableWidgetItem(QString::number(weights[w], 'f', 15));
    ui->tableWeights->insertRow(ui->tableWeights->rowCount());
    ui->tableWeights->setItem(ui->tableWeights->rowCount() - 1, 0, row);
  }
  loadingW = false;
}

void NNViewer::openNNConfigFile() {
  QString filePath = QFileDialog::getOpenFileName(this, "Open NN config file");

  cmNN::NeuralNetwork::loadConfiguration(filePath.toStdString().c_str(),
                                         &config);

  populateNNTree();
  populateCmbActivationFx();
}

void NNViewer::saveNNConfigFile() {
  QString filePath = QFileDialog::getSaveFileName(this, "Save NN config file");
  qDebug() << filePath;
}

void NNViewer::nnTreeAddRootNode(size_t layer,
                                 QString name,
                                 QString description) {
  QTreeWidgetItem* rootNode = new QTreeWidgetItem(ui->treeNN);

  rootNode->setText(0, name);
  rootNode->setText(1, description);

  for (size_t n = 0; n < config.neuronsPerLayer[layer]; n += 1) {
    size_t nW = getNumOfWeightsForNeuron(layer, n);
    size_t fW = getIndexFirstWeightForNeuron(layer, n);
    nnTreeAddChildNode(rootNode, QString::number(n), "", nW,
                       &config.weights[fW]);
  }
}

void NNViewer::nnTreeAddChildNode(QTreeWidgetItem* parent,
                                  QString name,
                                  QString description,
                                  size_t dim,
                                  double* data) {
  QTreeWidgetItem* childNode = new QTreeWidgetItem();
  bool isActive = data[dim - 2] > 0.0;
  QString iconPath =
      "/Users/jose/Documents/projects/DeepLearning_Tools/ui/NNViewer/assets/";
  iconPath = iconPath + (isActive ? "greenCircle.png" : "blueCircle.png");

  childNode->setText(0, name);
  childNode->setText(1, description);
  childNode->setIcon(1, QIcon(iconPath));

  parent->addChild(childNode);
}

void NNViewer::nnTreeUpdateChildNode(size_t layer, size_t neuron) {
  QTreeWidgetItem* n = ui->treeNN->topLevelItem(layer)->child(neuron);
  QString iconPath =
      "/Users/jose/Documents/projects/DeepLearning_Tools/ui/NNViewer/assets/";
  iconPath = iconPath + (isNeuronActive(layer, neuron) ? "greenCircle.png"
                                                       : "blueCircle.png");

  n->setText(0, n->text(0) + "  (updated)");
  n->setIcon(1, QIcon(iconPath));
}

void NNViewer::on_treeNN_itemSelectionChanged() {
  QList<QTreeWidgetItem*> selectedItems = ui->treeNN->selectedItems();
  if (selectedItems.count() > 0) {
    QTreeWidgetItem* item = selectedItems[0];
    QTreeWidgetItem* parent = item->parent();

    if (!parent) {
      size_t layer = ui->treeNN->indexFromItem(item, 0).row();
      populateLayerDetails(layer);
      return;
    }

    size_t layer = ui->treeNN->indexFromItem(parent, 0).row();
    size_t neuron = ui->treeNN->indexFromItem(item, 0).row();

    populateLayerDetails(layer);
    populateNeuronDetails(layer, neuron);
  }
}

void NNViewer::on_chkIsActive_clicked(bool checked) {
  setIsNeuronActive(cLayer, cNeuron, checked);
  nnTreeUpdateChildNode(cLayer, cNeuron);
}

void NNViewer::on_tableWeights_cellChanged(int row, int column) {
  if (loadingW)
    return;

  QTableWidgetItem* r = ui->tableWeights->item(row, column);
  setNeuronWeight(cLayer, cNeuron, row, r->text());
}

void NNViewer::on_btnDeleteNeuron_clicked() {
  config = deleteNeuron(cLayer, cNeuron);
  populateNNTree();
  populateCmbActivationFx();
}
