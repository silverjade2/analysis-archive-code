-- dbt Projects on Snowflake: dbt 폴더가 있는 GitHub 저장소를 Workspace 로 가져오기 위한 준비
-- 공개 저장소는 토큰 없이 가져올 수 있다. 비공개 저장소일 때만 아래 SECRET 이 필요
-- 순서: GitHub 에서 토큰 만들기 → 아래 <GITHUB_TOKEN> 자리에 붙여 넣고 실행 → Workspaces 에서 From Git repository
-- 토큰은 SECRET 객체 안에만 저장되고, 이 파일 (저장소) 에는 남기지 않는다. 실행 뒤 Workspaces 의 SQL 파일에서도 지울 것

USE ROLE ACCOUNTADMIN;
CREATE SCHEMA IF NOT EXISTS BIGDATA.INTEGRATIONS;

CREATE SECRET IF NOT EXISTS BIGDATA.INTEGRATIONS.GITHUB_PAT
  TYPE = PASSWORD
  USERNAME = 'silverjade2'
  PASSWORD = '<GITHUB_TOKEN>';

CREATE API INTEGRATION IF NOT EXISTS GITHUB_API
  API_PROVIDER = git_https_api
  API_ALLOWED_PREFIXES = ('https://github.com/silverjade2/')
  ALLOWED_AUTHENTICATION_SECRETS = (BIGDATA.INTEGRATIONS.GITHUB_PAT)
  ENABLED = TRUE;

-- 확인
SHOW SECRETS IN SCHEMA BIGDATA.INTEGRATIONS;
SHOW API INTEGRATIONS LIKE 'GITHUB_API';

-- dbt 를 돌린 뒤 결과 확인 (Workspace 에서 Build 실행 후)
-- SHOW TABLES IN SCHEMA BIGDATA.ANALYTICS_MARTS;
-- SELECT * FROM BIGDATA.ANALYTICS_MARTS.SKILL_DEMAND_VS_ACTIVITY ORDER BY POSTING_SHARE DESC;
