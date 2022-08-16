#ifndef NNVIEWER_H
#define NNVIEWER_H

#include <QMainWindow>
#include <QProperty>
#include <QTreeWidgetItem>
#include "cmDeepLearning.hpp"

QT_BEGIN_NAMESPACE
namespace Ui {
class NNViewer;
}
QT_END_NAMESPACE

class NNViewer : public QMainWindow {
  Q_OBJECT

 public:
  NNViewer(QWidget* parent = nullptr);
  ~NNViewer();

 private slots:

  void on_treeNN_itemSelectionChanged();
  void on_chkIsActive_clicked(bool checked);

  void on_tableWeights_cellChanged(int row, int column);

 private:
  Ui::NNViewer* ui;
  cmNN::NeuralNetworkConfiguration config;

  size_t cLayer = -1;
  size_t cNeuron = -1;
  bool loadingW = false;

  void nnTreeAddRootNode(size_t layer, QString name, QString description);
  void nnTreeAddChildNode(QTreeWidgetItem* parent,
                          QString name,
                          QString description,
                          size_t dim,
                          double* data);
  void nnTreeUpdateChildNode(size_t layer, size_t neuron);

  size_t getNumOfWeightsForNeuron(size_t layer, size_t neuron);
  size_t getIndexFirstWeightForNeuron(size_t layer, size_t neuron);
  bool isNeuronActive(size_t layer, size_t neuron);

  void setIsNeuronActive(size_t layer, size_t neuron, bool isActive);
  void setNeuronWeight(size_t layer,
                       size_t neuron,
                       size_t wIndex,
                       QString newWeight);

  void populateMainMenu();
  void populateCmbActivationFx();
  void populateNNTree();
  void populateLayerDetails(size_t layer);
  void populateNeuronDetails(size_t layer, size_t neuron);
  void openNNConfigFile();
};
#endif  // NNVIEWER_H
