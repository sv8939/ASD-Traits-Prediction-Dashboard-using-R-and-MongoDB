# Load Required Libraries
library(shiny)
library(shinydashboard)
library(mongolite)
library(ggplot2)
library(dplyr)
library(caret)
library(xgboost)
library(DT)
library(pROC)

# Define UI
ui <- dashboardPage(
  dashboardHeader(title = "ASD Traits Prediction"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Load Data", tabName = "data", icon = icon("database")),
      menuItem("EDA", tabName = "eda", icon = icon("chart-bar")),
      menuItem("Train Model", tabName = "model", icon = icon("rocket")),
      menuItem("Predict", tabName = "predict", icon = icon("magic"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "data",
              actionButton("load_data", "📥 Load Data"),
              DTOutput("data_table")
      ),
      tabItem(tabName = "eda",
              fluidRow(
                column(6, plotOutput("eda_plot_1")),
                column(6, plotOutput("eda_plot_2"))
              )
      ),
      tabItem(tabName = "model",
              actionButton("train_model", "🚀 Train Model"),
              verbatimTextOutput("accuracy_output"),
              plotOutput("roc_curve"),
              tableOutput("conf_matrix")
      ),
      tabItem(tabName = "predict",
              uiOutput("dynamic_inputs"),
              actionButton("predict_btn", "🔮 Predict ASD"),
              verbatimTextOutput("prediction_result")
      )
    )
  )
)

# Define Server Logic
server <- function(input, output, session) {
  df <- reactiveVal(NULL)
  xgb_model <- reactiveVal(NULL)
  feature_names <- reactiveVal(NULL)
  
  # Load Data from MongoDB
  observeEvent(input$load_data, {
    mongo_conn <- mongo(collection = "asd", db = "project", url = "mongodb://localhost:27017")
    data <- mongo_conn$find('{}')
    
    data$ASD_traits <- ifelse(data$ASD_traits == "Yes", 1, 0)
    
    df(data)
    feature_names(setdiff(names(data)[sapply(data, is.numeric)], "ASD_traits"))
    output$data_table <- renderDT({ datatable(data) })
  })
  
  # EDA
  output$eda_plot_1 <- renderPlot({
    if (is.null(df())) return()
    ggplot(df(), aes(x = factor(ASD_traits), fill = factor(ASD_traits))) +
      geom_bar() + theme_minimal() +
      labs(title = "Class Distribution", x = "ASD Traits", y = "Count") +
      scale_fill_manual(values = c("red", "blue"))
  })
  
  output$eda_plot_2 <- renderPlot({
    if (is.null(df())) return()
    feature_col <- feature_names()[1]
    ggplot(df(), aes(x = .data[[feature_col]], fill = factor(ASD_traits))) +
      geom_density(alpha = 0.6) + theme_minimal() +
      labs(title = "Feature Distribution", x = feature_col, y = "Density") +
      scale_fill_manual(values = c("red", "blue"))
  })
  
  # Model Training
  observeEvent(input$train_model, {
    if (is.null(df())) {
      output$accuracy_output <- renderPrint({ "❌ No data available! Load the data first." })
      return()
    }
    
    data <- df()
    features <- feature_names()
    
    set.seed(123)
    train_index <- createDataPartition(data$ASD_traits, p = 0.8, list = FALSE)
    train_data <- data[train_index, ]
    test_data  <- data[-train_index, ]
    
    train_matrix <- xgb.DMatrix(as.matrix(train_data[, features]), label = train_data$ASD_traits)
    test_matrix  <- xgb.DMatrix(as.matrix(test_data[, features]), label = test_data$ASD_traits)
    
    params <- list(
      objective = "binary:logistic",
      eval_metric = "error",
      max_depth = 6,
      eta = 0.1
    )
    
    model <- xgb.train(params = params, data = train_matrix, nrounds = 100)
    xgb_model(model)
    
    predictions <- predict(model, test_matrix)
    pred_labels <- ifelse(predictions > 0.5, 1, 0)
    
    accuracy <- mean(pred_labels == test_data$ASD_traits) * 100
    output$accuracy_output <- renderPrint({ paste("✅ Best Accuracy:", round(accuracy, 2), "%") })
    
    output$conf_matrix <- renderTable({
      table(Actual = test_data$ASD_traits, Predicted = pred_labels)
    })
    
    output$roc_curve <- renderPlot({
      roc_obj <- roc(test_data$ASD_traits, predictions)
      plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve for ASD Traits Prediction")
      abline(a = 0, b = 1, lty = 2, col = "red")
      legend("bottomright", legend = paste("AUC =", round(auc(roc_obj), 3)), col = "black", lwd = 2)
    })
  })
  
  # Dynamic Inputs for Prediction
  output$dynamic_inputs <- renderUI({
    req(feature_names())
    lapply(feature_names(), function(name) {
      numericInput(paste0("inp_", name), label = name, value = 0)
    })
  })
  
  # Prediction Logic
  observeEvent(input$predict_btn, {
    req(xgb_model())
    
    input_values <- sapply(feature_names(), function(name) input[[paste0("inp_", name)]])
    df_input <- matrix(as.numeric(input_values), nrow = 1)
    
    prediction <- predict(xgb_model(), xgb.DMatrix(df_input))
    result <- ifelse(prediction > 0.5, "🧠 High Possibility of ASD", "🙂 Low Possibility of ASD")
    
    output$prediction_result <- renderPrint({ paste("Prediction:", result) })
  })
}

# Run App
shinyApp(ui, server)
