#include "nnviewer.h"
#include "./ui_nnviewer.h"
#include <QAction>
#include <QMenu>
#include <QFileDialog>

#include "cmDeepLearning.hpp"

NNViewer::NNViewer(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::NNViewer)
{
    ui->setupUi(this);
    populateMainMenu();
}

NNViewer::~NNViewer()
{
    delete ui;
}

void NNViewer::populateMainMenu() {
    QMenu *fileMenu = ui->menubar->addMenu("File");
    QAction *openNNConfigAction = fileMenu->addAction("Open NN config file...");
    connect(openNNConfigAction, &QAction::triggered, this, &NNViewer::openNNConfigFile);
}

void NNViewer::openNNConfigFile() {
    QString filePath = QFileDialog::getOpenFileName(this, "Open NN config file");

    cmNN::NeuralNetworkConfiguration config;
    cmNN::NeuralNetwork::loadConfiguration(filePath.toStdString().c_str(), &config);
    ui->lblSomething->setText(QString::number(config.nInputs));
}
