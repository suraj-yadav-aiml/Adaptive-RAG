from configparser import ConfigParser, ParsingError

class Config:
    def __init__(self, config_file_path="./src/AdaptiveRag/ui/uiconfigfile.ini"):
        self.config = ConfigParser()
        
        try:
            with open(config_file_path, mode='r', encoding="utf-8-sig") as f:
                self.config.read_file(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at path: {config_file_path}")
        except PermissionError:
            raise ValueError(f"Permission denied accessing: {config_file_path}")
        except ParsingError as e:
            raise ParsingError(f"Error parsing config file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error reading config: {str(e)}")


        if "DEFAULT" not in self.config:
            raise ValueError("Missing 'DEFAULT' section in uiconfigfile.ini file")
    
    def get_llm_options(self):
        value = self.config['LLM_PROVIDERS'].get("PROVIDERS", "")
        return [llm.strip() for llm in value.split(",")] if value else []

    def get_groq_model_options(self):
        value = self.config['GROQ'].get('MODEL_OPTIONS', '')
        return [model.strip() for model in value.split(',')] if value else []

    def get_openai_model_options(self):
        value = self.config['OPENAI'].get('MODEL_OPTIONS', '')
        return [model.strip() for model in value.split(',')] if value else []
    
    def get_anthropic_model_options(self):
        value = self.config['ANTHROPIC'].get('MODEL_OPTIONS', '')
        return [model.strip() for model in value.split(',')] if value else []

    def get_page_title(self):
        return self.config['DEFAULT'].get('PAGE_TITLE', 'Adaptive RAG')

if __name__ == "__main__":
    config = Config()
    print(config.get_llm_options())
    print(config.get_groq_model_options())
    print(config.get_openai_model_options())
    print(config.get_anthropic_model_options())
    print(config.get_page_title())