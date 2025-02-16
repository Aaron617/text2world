PROJECT_NAME=${1}

if [ "$PROJECT_NAME" = "_all_gen" ]; then
    echo "Cannot create project with name '_all_gen'. Exiting..."
    exit 1
fi

echo "Creating new project: $PROJECT_NAME"
mkdir _generated_pddl/$PROJECT_NAME