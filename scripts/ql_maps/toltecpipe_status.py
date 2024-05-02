from dagster_graphql import DagsterGraphQLClient, DagsterGraphQLClientError
import sys
import yaml

RUNS_STARTED_QUERY = """
query FilteredRunsQuery {
  runsOrError(filter: { statuses: [STARTED] }) {
    __typename
    ... on Runs {
      results {
        runId
        jobName
        status
        runConfigYaml
        startTime
        endTime
      }
    }
  }
}
"""

def has_run_for_obsnum(client, obsnum):
    res_data = client._execute(
        RUNS_STARTED_QUERY
    )
    query_result = res_data["runsOrError"]
    query_result_type = query_result["__typename"]
    if query_result_type != "Runs":
        raise DagsterGraphQLClientError(query_result_type, query_result["message"])
    runs = query_result["results"]
    if not runs:
        print("no active runs")
        return False
    print(f"found {len(runs)} active runs")

    for run in runs:
        config = yaml.safe_load(run["runConfigYaml"])
        print(config)
        if config["ops"]["config"]["obsnum"] == obsnum:
            return True
    else:
        return False



def get_client():
    return DagsterGraphQLClient("localhost", port_number=3000)


if __name__ == "__main__":
    client = DagsterGraphQLClient("localhost", port_number=3000)
    print(client)
    obsnum = int(sys.argv[1])
    has_run = has_run_for_obsnum(client, obsnum)
    print(f"{obsnum=}: {has_run=}")
