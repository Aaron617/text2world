
import datetime
from typing import Literal, Optional


def get_openai_token(
    token_cache_file: str = 'cloudgpt-apim-token-cache.bin',
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> str:
    '''
    acquire token from Azure AD for CloudGPT OpenAI

    Parameters
    ----------
    token_cache_file : str, optional
        path to the token cache file, by default 'cloudgpt-apim-token-cache.bin' in the current directory
    client_id : Optional[str], optional
        client id for AAD app, by default None
    client_secret : Optional[str], optional
        client secret for AAD app, by default None

    Returns
    -------
    str
        access token for CloudGPT OpenAI
    '''
    import msal
    import os

    cache = msal.SerializableTokenCache()

    def save_cache():
        if token_cache_file is not None and cache.has_state_changed:
            with open(token_cache_file, "w") as cache_file:
                cache_file.write(cache.serialize())
    if os.path.exists(token_cache_file):
        cache.deserialize(open(token_cache_file, "r").read())

    authority = "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47"
    api_scope_base = "api://feb7b661-cac7-44a8-8dc1-163b63c23df2"

    if client_id is not None and client_secret is not None:
        app = msal.ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=authority,
            token_cache=cache
        )
        result = app.acquire_token_for_client(
            scopes=[
                api_scope_base + "/.default",
            ])
        if "access_token" in result:
            return result['access_token']
        else:
            print(result.get("error"))
            print(result.get("error_description"))
            raise Exception(
                "Authentication failed for acquiring AAD token for CloudGPT OpenAI")

    scopes = [api_scope_base + "/openai"]
    app = msal.PublicClientApplication(
        "feb7b661-cac7-44a8-8dc1-163b63c23df2",
        authority=authority,
        token_cache=cache
    )
    result = None
    for account in app.get_accounts():
        try:
            result = app.acquire_token_silent(scopes, account=account)
            if result is not None and "access_token" in result:
                save_cache()
                return result['access_token']
            result = None
        except Exception:
            continue

    accounts_in_cache = cache.find(msal.TokenCache.CredentialType.ACCOUNT)
    for account in accounts_in_cache:
        try:
            refresh_token = cache.find(
                msal.CredentialType.REFRESH_TOKEN,
                query={
                    "home_account_id": account["home_account_id"]
                })[0]
            result = app.acquire_token_by_refresh_token(
                refresh_token["secret"], scopes=scopes)
            if result is not None and "access_token" in result:
                save_cache()
                return result['access_token']
            result = None
        except Exception:
            pass

    if result is None:
        print("no token available from cache, acquiring token from AAD")
        # The pattern to acquire a token looks like this.
        flow = app.initiate_device_flow(scopes=scopes)
        print(flow['message'])
        result = app.acquire_token_by_device_flow(flow=flow)
        if result is not None and "access_token" in result:
            save_cache()
            return result['access_token']
        else:
            print(result.get("error"))
            print(result.get("error_description"))
            raise Exception(
                "Authentication failed for acquiring AAD token for CloudGPT OpenAI")


cloudgpt_available_models = Literal[
    "gpt-35-turbo-20220309",
    "gpt-35-turbo-16k-20230613",
    "gpt-35-turbo-20230613",
    "gpt-35-turbo-1106",

    "gpt-4-20230321",
    "gpt-4-20230613",
    "gpt-4-32k-20230321",
    "gpt-4-32k-20230613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    
    "gpt-4-visual-preview",
]


def encode_image(image_path: str, mime_type: Optional[str] = None) -> str:
    import base64
    import os
    import mimetypes
    file_name = os.path.basename(image_path)
    mime_type = mime_type if mime_type is not None else mimetypes.guess_type(file_name)[0]
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('ascii')
        
    if mime_type is None or not mime_type.startswith("image/"):
        print("Warning: mime_type is not specified or not an image mime type. Defaulting to png.")
        mime_type = "image/png"
        
    image_url = f"data:{mime_type};base64," + encoded_image
    return image_url

def get_chat_completion(
    model: cloudgpt_available_models = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    *args,
    **kwargs
):
    """
    helper function for getting chat completion from CloudGPT OpenAI
    """
    import openai

    model_name: str = \
        model if model is not None else \
        kwargs.get("engine") if kwargs.get("engine") is not None else \
        None
    
    if "engine" in kwargs:
        del kwargs["engine"]
    
    client = openai.AzureOpenAI(
        api_version="2023-12-01-preview",
        azure_endpoint="https://cloudgpt-openai.azure-api.net/",
        azure_ad_token=get_openai_token(client_id=client_id, client_secret=client_secret),
    )

    response = client.chat.completions.create(
        model=model_name,
        *args,
        **kwargs
    )
    
    return response


def auto_refresh_token(
    token_cache_file: str = 'cloudgpt-apim-token-cache.bin',
    interval: datetime.timedelta = datetime.timedelta(minutes=15),
    on_token_update: callable = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> callable:
    """
    helper function for auto refreshing token from CloudGPT OpenAI

    Parameters
    ----------
    token_cache_file : str, optional
        path to the token cache file, by default 'cloudgpt-apim-token-cache.bin' in the current directory
    interval : datetime.timedelta, optional
        interval for refreshing token, by default 15 minutes
    on_token_update : callable, optional
        callback function to be called when token is updated, by default None. In the callback function, you can get token from openai.api_key

    Returns
    -------
    callable
        a callable function that can be used to stop the auto refresh thread
    """

    import threading

    def update_token():
        import openai

        openai.api_type = "azure"
        openai.base_url = "https://cloudgpt-openai.azure-api.net/"
        openai.api_version = "2023-12-01-preview"
        openai.api_key = get_openai_token(
            token_cache_file=token_cache_file,
            client_id=client_id,
            client_secret=client_secret,
        )

        if on_token_update is not None:
            on_token_update()

    def refresh_token_thread():
        import time
        while True:
            try:
                update_token()
            except Exception as e:
                print("failed to acquire token from AAD for CloudGPT OpenAI", e)
            time.sleep(interval.total_seconds())

    try:
        update_token()
    except Exception as e:
        raise Exception(
            "failed to acquire token from AAD for CloudGPT OpenAI", e)

    thread = threading.Thread(target=refresh_token_thread, daemon=True)
    thread.start()

    def stop():
        thread.stop()

    return stop


def test_get_chat_completion():
    def test_call(*args, **kwargs):
        test_message = "What is the content?"
        test_chat_message = [{"role": "user", "content": test_message}]

        response = get_chat_completion(
            engine="gpt-35-turbo-20220309",
            messages=test_chat_message,
            temperature=0.7,
            max_tokens=100,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            *args,
            **kwargs
        )

        print(response.choices[0].message)

    print("test without AAD app")
    test_call()  # test without AAD app

    import os
    client_id = os.environ.get('CLIENT_ID')
    client_secret = os.environ.get('CLIENT_SECRET')

    if client_id is None or client_secret is None:
        print("please set CLIENT_ID and CLIENT_SECRET environment variables for testing AAD app")
    else:
        print("test with AAD app")
        # test with AAD app
        test_call(client_id=client_id, client_secret=client_secret)


if __name__ == "__main__":
    test_get_chat_completion()
