pipeline{
    agent any

    environment{
        VENV_DIR = 'venv'
    }

    stages{

        stage('Cloning github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning github repo to Jenkins ...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github_token', url: 'https://github.com/Nikelroid/hotel_cancel_detection-MLOps']])
                }
            }
        }

        stage('Setting up virtual environment and installing dependencies'){
            steps{
                script{
                    echo 'Setting up virtual environment and installing dependencies ...'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
    }
}